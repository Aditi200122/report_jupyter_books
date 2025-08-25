# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# class DonationAnalyzer:
#     def __init__(self, donations_df, messages_df):
#         self.donations = donations_df
#         self.messages = messages_df
#         self.parsers = self._group_by_donations()

#     def _group_by_donations(self):
#         grouped = []
#         for donation_id, group in self.messages.groupby("donation_id"):
#             grouped.append(DonationParser(donation_id, group))
#         return grouped

#     def get_all_donors(self):
#         return list(self.donations["donor_id"].unique())

#     def aggregate_counts(self, donor_id, metric="Messages"):
#         counts = {}
#         for parser in self.parsers:
#             contact, msg_count, word_count = parser.count_donor_msgs_words(donor_id)
#             if contact is not None:
#                 counts[contact] = msg_count if metric == "Messages" else word_count
#         return counts

#     def calculate_gini(self, counts):
#         values = sorted(counts.values())
#         n = len(values)
#         total = sum(values)
#         if n == 0 or total == 0:
#             return 0.0
#         weighted_sum = sum((i + 1) * val for i, val in enumerate(values))
#         return (2 * weighted_sum) / (n * total) - (n + 1) / n

#     def plot_lorenz_curve(self, counts, donor_name=""):
#         values = np.array(sorted(counts.values()))
#         if len(values) == 0:
#             print("No data to plot.")
#             return

#         cumulative = np.cumsum(values) / values.sum()
#         cumulative = np.insert(cumulative, 0, 0)
#         contacts = np.linspace(0, 1, len(values) + 1)

#         gini = self.calculate_gini(counts)

#         plt.figure(figsize=(4, 4))
#         plt.plot(contacts * 100, cumulative * 100, label="Lorenz Curve", color="blue")
#         plt.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Perfect Equality")
#         plt.fill_between(contacts * 100, contacts * 100, cumulative * 100, color="lightblue", alpha=0.3)
#         plt.title(f"Lorenz Curve (Gini = {gini:.3f})")
#         plt.xlabel("Cumulative % of Contacts")
#         plt.ylabel("Cumulative % of Words or Messages")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

# class DonationParser:
#     def __init__(self, donation_id, df):
#         self.donation_id = donation_id
#         self.df = df

#     def count_donor_msgs_words(self, donor_id):
#         donor_msgs = self.df[self.df["sender_id"] == donor_id]
#         other_msgs = self.df[self.df["sender_id"] != donor_id]

#         if donor_msgs.empty or other_msgs.empty:
#             return None, 0, 0

#         contact_id = other_msgs["sender_id"].unique()[0]
#         msg_count = donor_msgs.shape[0]
#         word_count = donor_msgs["word_count"].sum()

#         return contact_id, msg_count, word_count
    

# class DonorBurstinessAnalyzer:
#     def __init__(self, messages_df, donations_df):
#         self.messages_df = messages_df
#         self.donations_df = donations_df

#     def plot_burstiness_distribution(self, donor_id):
#         donor_msgs = self.messages_df[self.messages_df['donation_id'].isin(
#             self.donations_df[self.donations_df['donor_id'] == donor_id]['donation_id']
#         )]

#         #messages sent by the donor with timestamps
#         donor_msgs = donor_msgs[donor_msgs['sender_id'] == donor_id]
#         donor_msgs = donor_msgs.sort_values("datetime")

#         #Convertign timestamp (WhatsApp is in device time) to pandas datetime
#         donor_msgs["datetime"] = pd.to_datetime(donor_msgs["datetime"], errors="coerce")

#         #Droping rows where datetime couldn't be parsed
#         donor_msgs = donor_msgs.dropna(subset=["datetime"])

#         #Calculating time differences in seconds between consecutive messages
#         time_deltas = donor_msgs["datetime"].diff().dt.total_seconds().dropna()

#         if len(time_deltas) < 2:
#             print(" Not enough data to calculate burstiness.")
#             return 0, 0

#         mean = np.mean(time_deltas)
#         std = np.std(time_deltas)

#         burstiness = (std - mean) / (std + mean) if (std + mean) > 0 else 0

#         #histogram
#         plt.figure(figsize=(6, 4))
#         plt.hist(time_deltas, bins=30, color="purple", alpha=0.7)
#         plt.title(f"Message Gaps for Donor {donor_id}")
#         plt.xlabel("Time gap between messages (seconds)")
#         plt.ylabel("Frequency")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#         return round(mean, 2), round(std, 2)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DONATION_CSV = r"C:/Users/Dev/Documents/GitHub/report_jupyter_books/real_data/12570525/donation_table.csv"
MESSAGES_CSV = r"C:/Users/Dev/Documents/GitHub/report_jupyter_books/real_data/12570525/messages_filtered_table.csv"

donations = pd.read_csv(DONATION_CSV)
donations = donations[donations["source"] == "WhatsApp"]

messages = pd.read_csv(MESSAGES_CSV)
messages = messages[messages["donation_id"].isin(donations["donation_id"])]

#Gini and Lorenz 
def calculate_gini(counts):
    values = sorted(counts.values())
    n = len(values)
    total = sum(values)
    if n == 0 or total == 0:
        return 0
    weighted_sum = sum((i + 1) * val for i, val in enumerate(values))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n

def plot_lorenz_curve(counts, title):
    values = np.array(sorted(counts.values()))
    cumulative = np.cumsum(values) / values.sum()
    cumulative = np.insert(cumulative, 0, 0)
    contacts = np.linspace(0, 1, len(values) + 1)

    gini = calculate_gini(counts)

    plt.figure(figsize=(6, 5))
    plt.plot(contacts * 100, cumulative * 100, label='Lorenz Curve', color='blue')
    plt.plot([0, 100], [0, 100], linestyle='--', color='gray', label='Perfect Equality')
    plt.fill_between(contacts * 100, contacts * 100, cumulative * 100, color='lightblue', alpha=0.3)
    plt.title(f"{title} (Gini Index = {gini:.3f})")
    plt.xlabel("Cumulative % of Contacts")
    plt.ylabel("Cumulative % of Messages/Words")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Burstiness
def compute_burstiness(days):
    days_sorted = sorted(days)
    inter_event_times = np.diff(pd.to_datetime(days_sorted)).astype('timedelta64[D]').astype(int)
    if len(inter_event_times) == 0:
        return None, None
    mu = np.mean(inter_event_times)
    sigma = np.std(inter_event_times)
    if mu == 0:
        return None, None
    r = sigma / mu
    n = len(days_sorted)
    B1 = (r - 1) / (r + 1) if (r + 1) != 0 else None
    if n > 1:
        numerator = (np.sqrt(n + 1) * r) - np.sqrt(n - 1)
        denominator = ((np.sqrt(n + 1) - 2) * r) + np.sqrt(n - 1)
        B2 = numerator / denominator if denominator != 0 else None
    else:
        B2 = None
    return B1, B2

def plot_burstiness_dashboard(donor_msgs, donor):
    donor_msgs = donor_msgs[donor_msgs['sender_id'] == donor].copy()

    
    donor_msgs['datetime'] = pd.to_datetime(
        donor_msgs['datetime'],
        format='mixed',
        errors='coerce'
    )
    donor_msgs = donor_msgs.dropna(subset=['datetime'])
    donor_msgs['date_only'] = donor_msgs['datetime'].dt.date

    interaction_days = donor_msgs.groupby('conversation_id')['date_only'].apply(lambda x: sorted(set(x)))
    interaction_days = interaction_days[interaction_days.apply(len) >= 10]

    if interaction_days.empty:
        print("No chats with ≥10 interaction days for burstiness analysis.")
        return

    burstiness_results = interaction_days.apply(lambda days: compute_burstiness(days))
    burstiness_df = pd.DataFrame(
        burstiness_results.tolist(),
        index=interaction_days.index,
        columns=['B1', 'B2']
    ).dropna()

    conv_id = burstiness_df['B1'].abs().idxmax()
    B1_extreme = burstiness_df.loc[conv_id, 'B1']
    B2_extreme = burstiness_df.loc[conv_id, 'B2']

    if B1_extreme > 0.5:
        donor_type = "Bursty"
        color = "red"
    elif B1_extreme < -0.5:
        donor_type = "Regular"
        color = "green"
    else:
        donor_type = "Random"
        color = "blue"

    days = interaction_days[conv_id]

    plt.figure(figsize=(8, 2))
    plt.eventplot(pd.to_datetime(sorted(days)), orientation='horizontal', colors=color)
    plt.title(f"{donor_type} chat (Conversation {conv_id})\nB1={B1_extreme:.2f}, B2={B2_extreme:.2f}")
    plt.xlabel("Time")
    plt.yticks([])
    plt.grid(axis='x')
    plt.show()
