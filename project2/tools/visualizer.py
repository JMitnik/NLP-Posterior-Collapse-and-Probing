# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
va = 'validation_accs'
cva = 'corrupted_validation_accs'

gpt2_va = 'GPT2 Original'
gpt2_cva = 'GPT2 Ctrl.'

xlm_va = 'X.R Original'
xlm_cva = 'X.R. Ctrl.'

lstm_va = 'LSTM Original'
lstm_cva = 'LSTM Ctrl.'

# %%
f = plt.figure(figsize=(6,6))
gs = f.add_gridspec(3,1)


# %%
# Load in all POS data:
POS_gpt2_df = pd.read_csv(f'../results/POS_probe/pos-training-valid/pos-distilgpt2.csv')
POS_gpt2 = POS_gpt2_df[[va, cva]].copy()
POS_gpt2 = POS_gpt2.rename(columns={va:gpt2_va, cva:gpt2_cva})

POS_xlm_df = pd.read_csv(f'../results/POS_probe/pos-training-valid/pos-xlm-roberta.csv')
POS_xlm = POS_xlm_df[[va, cva]].copy()
POS_xlm = POS_xlm.rename(columns={va:xlm_va, cva:xlm_cva})

POS_lstm_df = pd.read_csv(f'../results/POS_probe/pos-training-valid/pos-lstm.csv')
POS_lstm = POS_lstm_df[[va, cva]].copy()
POS_lstm = POS_lstm.rename(columns={va:lstm_va, cva:lstm_cva})

# %%
# Plot training validation for POS probe
sns.set_context("paper", font_scale=1.3)
sns.set_style("darkgrid")

sns.set_palette(sns.color_palette("BuGn_r"))
ax1 = sns.lineplot(data=POS_gpt2, alpha=0.7)

sns.set_palette(sns.light_palette("navy", reverse=True))
ax2 = sns.lineplot(data=POS_lstm, alpha=0.7)

sns.set_palette(sns.color_palette("BrBG", 7))
ax2 = sns.lineplot(data=POS_xlm, alpha=0.7)

plt.title('POS Probe')
plt.xlabel('epochs')
plt.ylabel('accuracy')

# %%
# Load in all Struct data:
Struct_gpt2_df = pd.read_csv(f'../results/Structural_probe/struct-training-valid/struct-distilgpt2.csv')
Struct_gpt2 = Struct_gpt2_df[['probe_valid_uuas_scores']].copy()
Struct_gpt2 = Struct_gpt2.rename(columns={'probe_valid_uuas_scores': 'GPT-2'})

Struct_xlm_df = pd.read_csv(f'../results/Structural_probe/struct-training-valid/struct-xlmroberta.csv')
Struct_xlm = Struct_xlm_df[['probe_valid_uuas_scores']].copy()
Struct_xlm = Struct_xlm.rename(columns={'probe_valid_uuas_scores': 'X.R.'})

Struct_lstm_df = pd.read_csv(f'../results/Structural_probe/struct-training-valid/struct-lstm.csv')
Struct_lstm = Struct_lstm_df[['probe_valid_uuas_scores']].copy()
Struct_lstm = Struct_lstm.rename(columns={'probe_valid_uuas_scores': 'LSTM'})

# %%
sns.set_context("paper", font_scale=1.3)
sns.set_style("darkgrid")

sns.set_palette(sns.color_palette("BuGn_r"))
ax1 = sns.lineplot(data=Struct_gpt2, alpha=0.7)

sns.set_palette(sns.light_palette("navy", reverse=True))
ax2 = sns.lineplot(data=Struct_lstm, alpha=0.7)

sns.set_palette(sns.color_palette("BrBG", 7))
ax2 = sns.lineplot(data=Struct_xlm, alpha=0.7)

plt.title('Structural Probe')
plt.xlabel('epochs')
plt.ylabel('UUAS')

# %%
# Load in all Edge data:
edge_gpt2_df = pd.read_csv(f'../results/Edge_probe/dep-training-valid/dep_gpt2.csv')
edge_gpt2 = edge_gpt2_df[['valid_acc', 'corrupted_dep_valid_acc']].copy()
edge_gpt2 = edge_gpt2.rename(columns={'valid_acc':gpt2_va, 'corrupted_dep_valid_acc':gpt2_cva})

edge_xlm_df = pd.read_csv(f'../results/Edge_probe/dep-training-valid/dep_xlm_roberta.csv')
edge_xlm = edge_xlm_df[['valid_acc', 'corrupted_dep_valid_acc']].copy()
edge_xlm = edge_xlm.rename(columns={'valid_acc':xlm_va, 'corrupted_dep_valid_acc':xlm_cva})

edge_lstm_df = pd.read_csv(f'../results/Edge_probe/dep-training-valid/dep_lstm.csv')
edge_lstm = edge_lstm_df[['valid_acc', 'corrupted_dep_valid_acc']].copy()
edge_lstm = edge_lstm.rename(columns={'valid_acc':lstm_va, 'corrupted_dep_valid_acc':lstm_cva})

print(gpt2_cva)
edge_gpt2.head()

# %%
# Plot training validation for Dependecy probe
sns.set_context("paper", font_scale=1.3)
sns.set_style("darkgrid")

sns.set_palette(sns.color_palette("BuGn_r"))
ax1 = sns.lineplot(data=edge_gpt2, alpha=0.7)

sns.set_palette(sns.light_palette("navy", reverse=True))
ax2 = sns.lineplot(data=edge_lstm, alpha=0.7)

sns.set_palette(sns.color_palette("BrBG", 7))
ax2 = sns.lineplot(data=edge_xlm, alpha=0.7)

plt.title('Dependency Edge Probe')
plt.xlabel('epochs')
plt.ylabel('accuracy')

# %%


# %%
