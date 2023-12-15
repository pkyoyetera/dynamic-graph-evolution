#
# Patrick Kyoyetera, Vijay Gochinghar
# 2023
#

from platform import python_version

print(f"Using Python{python_version()}")

import torch
import torch.optim as optim

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from model import RecommenderSystem
from utils.utils import compute_bpr_loss, data_loader, get_metrics
from utils.parser import parse_args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

print(f"Loading data and preprocessing...")
column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=column_names)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_df = pd.DataFrame(train_data, columns=column_names)
test_df = pd.DataFrame(test_data, columns=column_names)

print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

# Create user and item label encoders, and relabel from 0
user_le, item_le = LabelEncoder(), LabelEncoder()

train_df["user_id_index"] = user_le.fit_transform(train_df["user_id"].values)
train_df["item_id_index"] = item_le.fit_transform(train_df["item_id"].values)

train_user_ids = train_df["user_id"].unique()
train_item_ids = train_df["item_id"].unique()

test_df = test_df[
    (test_df["user_id"].isin(train_user_ids))
    & (test_df["item_id"].isin(train_item_ids))
]
print(f"Test size after rebalancing: {len(test_df)}")

test_df["user_id_index"] = user_le.transform(test_df["user_id"].values)
test_df["item_id_index"] = item_le.transform(test_df["item_id"].values)

n_users = train_df["user_id_index"].nunique()
n_items = train_df["item_id_index"].nunique()

print("=" * 80)

print(f"Number of unique users: {n_users}")
print(f"Number of unique items: {n_items}")
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

print(f"Done processing train and test data")

u_t = torch.LongTensor(train_df.user_id_index)
i_t = torch.LongTensor(train_df.item_id_index) + n_users

train_edge_index = torch.stack((torch.cat([u_t, i_t]), torch.cat([i_t, u_t]))).to(
    device
)


if __name__ == "__main__":
    args = parse_args()

    epochs = args.n_epochs
    dim = args.emb_dim
    num_layers = args.layers
    batch_size = args.batch_size
    learning_rate = args.lr
    k = args.k
    drop = args.dropout

    decay = 1e-4
    global_counter = 0

    arch = args.arch
    model = RecommenderSystem(dim, num_layers, n_users, n_items, conv_name=arch, dropout=drop)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        n_batch = int(len(train_df) / batch_size)

        model.train()
        for batch_idx in range(n_batch):
            optimizer.zero_grad()

            users, pos_items, neg_items = data_loader(
                train_df, batch_size, n_users, n_items
            )

            (
                users_emb,
                pos_emb,
                neg_emb,
                userEmb0,
                posEmb0,
                negEmb0,
            ) = model.encode_minibatch(users, pos_items, neg_items, train_edge_index)

            bpr_loss, reg_loss = compute_bpr_loss(
                users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0
            )
            reg_loss *= decay
            final_loss = bpr_loss + reg_loss

            final_loss.backward()
            optimizer.step()

            global_counter += 1

        model.eval()
        with torch.no_grad():
            _, out = model(train_edge_index)

            final_user_embed, final_item_embed = torch.split(out, (n_users, n_items))

            test_topK_recall, test_topK_precision = get_metrics(
                final_user_embed,
                final_item_embed,
                n_users,
                n_items,
                train_df,
                test_df,
                k,
                device,
            )

            print(f"Evaluating at epoch {epoch}")
            print(f"Test Recall@{k}: {test_topK_recall}")
            print(f"Test Precision@{k}: {test_topK_precision}")
            print("=" * 80)

    print(f"Done training recommender model with {arch} architecture for {epochs} epochs")
    print(f"Final precision@{k}: {test_topK_precision}")
    print(f"Final recall@{k}: {test_topK_recall}")
