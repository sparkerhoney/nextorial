import pandas as pd

class DataProcessor:
    
    def __init__(self, df):
        self.data = df

    @staticmethod
    def convert_tier_to_numeric(tier):
        tier_dict = {
            'unranked': 0,
            'bronze': 1,
            'silver': 2,
            'gold': 3,
            'platinum': 4,
            'diamond': 5,
            'master': 6
        }
        return tier_dict.get(tier, -1)

    def calculate_team_features(self):
        return self.data.groupby(['matchid', 'teamid']).agg({
            'mmr': 'mean',
            'winstreak': 'mean',
            'losestreak': 'mean',
            'recentwinprob': 'mean'
        }).reset_index()

    def preprocess(self):
        match_team_features = self.calculate_team_features()
        self.data = self.data.merge(match_team_features, on=['matchid', 'teamid'], suffixes=('', '_team_avg'))
        self.data = self.guild_avg()
        self.data = self.normalize_column('accumatches')
        self.data = self.compute_team_stats()
        self.data = self.compute_recent_performance_index()
        self.data = self.process_guild_info(2000)
        self.data = self.compute_mmr_diff_and_variance()
        self.data = self.compute_recent_winprob_stats()
        self.data = self.apply_tier_conversion_and_compute_average()
        self.data = self.compute_streak_rate()

    def guild_avg(self):
        df = self.data.copy()
        df['guildid'].fillna('NoGuild', inplace=True)
        df['tier_numeric'] = df['tier'].map(self.convert_tier_to_numeric)
        guild_avg = df.groupby('guildid')[['mmr', 'winstreak', 'recentwinprob', 'accumatches', 'tier_numeric']].mean()
        new_columns = {col: f'{col}_guild_avg' for col in guild_avg.columns}
        guild_avg.rename(columns=new_columns, inplace=True)
        df = pd.merge(df, guild_avg, on='guildid', how='left')
        return df

    def normalize_column(self, column):
        self.data[f'normalized_{column}'] = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())
        return self.data

    def compute_team_stats(self):
        grouped = self.data.groupby(['matchid', 'teamid'])
        self.data['team_max_accumatches'] = grouped['accumatches'].transform('max')
        self.data['team_min_accumatches'] = grouped['accumatches'].transform('min')
        self.data['accumatches_diff'] = self.data['team_max_accumatches'] - self.data['team_min_accumatches']
        self.data['accumatches_variance'] = grouped['accumatches'].transform('var')
        return self.data

    def compute_recent_performance_index(self):
        self.data['recent_performance_index'] = self.data['winstreak'] * self.data['recentwinprob']
        return self.data

    def process_guild_info(self, threshold):
        guild_mean_mmr = self.data.groupby('guildid')['mmr'].mean()
        self.data['guild_mean_mmr'] = self.data['guildid'].map(guild_mean_mmr)
        self.data['high_mmr_guild'] = (self.data['guild_mean_mmr'] > threshold).astype(int)
        return self.data

    def compute_mmr_diff_and_variance(self):
        mmr_diff_grouped = self.data.groupby('teamid')['mmr'].agg(['max', 'min'])
        self.data['mmr_diff'] = self.data['teamid'].map(mmr_diff_grouped['max'] - mmr_diff_grouped['min'])
        mmr_variance_grouped = self.data.groupby('teamid')['mmr'].var()
        self.data['mmr_variance'] = self.data['teamid'].map(mmr_variance_grouped)
        return self.data

    def compute_recent_winprob_stats(self):
        grouped = self.data.groupby('matchid')
        self.data['recentwinprob_max'] = grouped['recentwinprob'].transform('max')
        self.data['recentwinprob_min'] = grouped['recentwinprob'].transform('min')
        self.data['recentwinprob_diff'] = self.data['recentwinprob_max'] - self.data['recentwinprob_min']
        self.data['recentwinprob_mean'] = grouped['recentwinprob'].transform('mean')
        self.data['recentwinprob_diff_from_mean'] = (self.data['recentwinprob'] - self.data['recentwinprob_mean'])**2
        self.data['recentwinprob_variance'] = grouped['recentwinprob_diff_from_mean'].transform('mean')
        return self.data

    def apply_tier_conversion_and_compute_average(self):
        self.data['tier_numeric'] = self.data['tier'].apply(self.convert_tier_to_numeric)
        average_tier = self.data.groupby(['matchid', 'teamid'])['tier_numeric'].mean().reset_index()
        average_tier.rename(columns={'tier_numeric': 'average_tier'}, inplace=True)
        self.data = self.data.merge(average_tier, on=['matchid', 'teamid'])
        return self.data

    @staticmethod
    def calculate_streak_rate(row):
        winstreak, losestreak = row['winstreak'], row['losestreak']
        if winstreak + losestreak == 0:
            return 0
        return winstreak / (winstreak + losestreak)

    def compute_streak_rate(self):
        self.data['streak_rate'] = self.data.apply(self.calculate_streak_rate, axis=1)
        return self.data