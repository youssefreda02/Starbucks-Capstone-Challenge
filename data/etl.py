import sys
import pandas as pd
from collections import Counter

from functions import user_offers_response 


def load_data(portfolio_filepath, profile_filepath, transcript_filepath):
    """
    Load offers, users, and transcript datasets.

    Args:
    messages_filepath: str. Path to the messages JSON file.
    categories_filepath: str. Path to the categories JSON file.

    Returns:
    : pandas DataFrames. Each data frame with each json file.
    """
    # Load messages dataset
    portfolio = pd.read_json(portfolio_filepath, orient='records', lines=True)
    profile = pd.read_json( profile_filepath, orient='records', lines=True)
    transcript = pd.read_json(transcript_filepath, orient='records', lines=True)
    
    return portfolio, profile, transcript


def clean_data(portfolio, profile, transcript):
    """
        Clean and preprocess the portfolio, profile, and transcript datasets for further analysis.

        This function performs a series of cleaning steps on the input datasets, including:
        - Removing missing values and outliers from the profile data.
        - Dropping corresponding events in the transcript where the associated profile is missing.
        - Flattening nested columns in the transcript data and normalizing offer information.
        - Handling informational offers by creating inferred "offer completed" events.
        - Merging counts of completed offers back into the portfolio data.
        - Expanding the `channels` column into separate columns for each channel.
        - Counting and merging user interaction metrics with the profile data.

        Parameters:
        - portfolio (pd.DataFrame): The portfolio dataset containing details of offers, including type, duration, and difficulty.
        - profile (pd.DataFrame): The profile dataset containing user demographic information such as age and gender.
        - transcript (pd.DataFrame): The transcript dataset containing records of events related to offers and transactions.

        Returns:
        - tuple: A tuple containing cleaned and processed versions of the profile, portfolio, and transcript datasets.

        Steps:
        1. Remove users with missing gender from the profile dataset, and filter related events from the transcript.
        2. Drop outliers from the profile dataset based on the age attribute.
        3. Normalize and concatenate offer data from nested JSON in the transcript.
        4. Infer "offer completed" events for informational offers if viewed and transactions are made within the offer duration.
        5. Merge offer completion counts with the portfolio, expanding the channels column.
        6. Count and merge user-specific event interactions (received, viewed, completed) into the profile dataset.

    """
    
    ids_na = profile[profile['gender'].isna()]['id']
    # First drop na
    profile_clean = profile.copy() # intial copy
    profile_clean.dropna(axis = 0, inplace = True)

    #Drop these people events
    transcript_clean = transcript.copy()# intial copy
    transcript_clean = transcript_clean[~transcript_clean['person'].isin(ids_na)]

    #After dropping the NANs, let's drop the outliers

    Q3 = profile_clean['age'].quantile(0.75)
    Q1 = profile_clean['age'].quantile(0.25)
    IQR = Q3 - Q1
    highest_threshold = 1.5* IQR + Q3
    profile_clean = profile_clean[profile_clean['age'] < highest_threshold]

    norma = pd.json_normalize(transcript_clean['value']).rename(columns={'offer id': 'offer_id_1', 'offer_id': 'offer_id_2'})

    # Reset indices before concatenation to avoid misalignment
    transcript_clean.reset_index(drop=True, inplace=True)
    norma.reset_index(drop=True, inplace=True)

    # Concatenate the original dataframe without the 'value' column with the normalized dataframe
    both = pd.concat([transcript_clean.drop('value', axis=1), norma], axis=1)

    # Create a single 'offer_id' column, using values from 'offer_id_1' and 'offer_id_2'
    both['offer_id'] = both['offer_id_1'].combine_first(both['offer_id_2'])

    # Drop the redundant columns
    transcript_clean = both.drop(columns=['offer_id_1', 'offer_id_2'])
    transcript_clean = transcript_clean.fillna(0)

    # Get the list of informational offers with their duration
    informational_offers = portfolio[portfolio['offer_type'] == 'informational'][['id', 'duration']]

    # Sort transcript_clean by user and time to maintain a chronological order
    # Create a copy to avoid modifying the original DataFrame directly
    transcript_updated = transcript_clean.sort_values(by=['person', 'time']).reset_index(drop=True).copy()

    # Iterate over users
    for user_id in transcript_clean['person'].unique():
        # Get all events related to the user
        user_events = transcript_clean[transcript_clean['person'] == user_id]
    
        # Iterate over each informational offer
        for _, row in informational_offers.iterrows():
            offer_id = row['id']
            offer_duration = row['duration'] * 24 

            # Find the "offer received" event for the current offer
            received_events = user_events[(user_events['offer_id'] == offer_id) & (user_events['event'] == 'offer received')]
        
            if not received_events.empty:
                # Get the time of the last "offer received" event
                received_time = received_events['time'].values[-1]
                offer_expiry_time = received_time + offer_duration

                # Check if there was a "viewed" event within the offer duration
                viewed_events = user_events[(user_events['offer_id'] == offer_id) & 
                                            (user_events['event'] == 'offer viewed') & 
                                            (user_events['time'] <= offer_expiry_time)]
            
                if not viewed_events.empty:
                    # Get the time of the last viewing event
                    viewed_time = viewed_events['time'].values[-1]

                    # Check for a subsequent transaction within the offer duration
                    transaction_events = user_events[(user_events['event'] == 'transaction') & 
                                                     (user_events['time'] > viewed_time) & 
                                                     (user_events['time'] <= offer_expiry_time)]
                
                    if not transaction_events.empty:
                        # Get the time of the transaction
                        transaction_time = transaction_events['time'].values[0]

                        # Create a new row representing "offer completed"
                        new_row = {
                            'person': user_id,
                            'event': 'offer completed',
                            'time': transaction_time + 1,  # Adding a small time increment to insert after transaction
                            'offer_id': offer_id,
                        }

                        # Append the new row to the updated DataFrame
                        transcript_updated = pd.concat([transcript_updated, pd.DataFrame([new_row])], ignore_index=True)

    # Sort the updated DataFrame to maintain chronological order
    transcript_updated = transcript_updated.sort_values(by=['time','person']).reset_index(drop=True)

    transcript_clean = transcript_updated

    count_merged_df = portfolio.merge(transcript_clean[transcript_clean['event'] == 'offer completed'].value_counts('offer_id'), left_on= 'id', right_on= 'offer_id')
    # Step 1: Create x-axis labels by combining 'offer_type', 'duration', and 'difficulty'
    count_merged_df['x_axis'] = count_merged_df['offer_type'] + ' in ' + count_merged_df['duration'].astype(str) + ' days limited by $' + count_merged_df['difficulty'].astype(str)

    # Filter the DataFrame to only include rows with a non-zero count
    filtered_df = count_merged_df[count_merged_df['count'] != 0]

    channels_df = count_merged_df['channels'].apply(lambda x: pd.Series({ch: 1 for ch in x})).fillna(0).astype(int)

    # Step 2: Rename the columns for clarity
    channels_df.columns = [f'channel_{col}' for col in channels_df.columns]

    # Step 3: Concatenate the expanded 'channels' with the original dataframe
    merged_df_expanded = pd.concat([count_merged_df.drop(columns=['channels', 'count','x_axis']), channels_df], axis=1)
    portfolio_clean = merged_df_expanded

    # Next, adding more columns in the profile_clean dataframe to help understand how many offers each user has completed, viewed, or just received.

    # Step 1: Filter the transcript for relevant events and count each event by person
    event_counts = (
        transcript_clean[transcript_clean['event'].isin(['offer received', 'offer viewed', 'offer completed', 'transaction'])]
        .groupby(['person', 'event'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Step 2: Rename columns for clarity
    event_counts.columns = ['person', 'offer_completed', 'offer_received', 'offer_viewed', 'transactions']

    # Step 3: Merge with profile_clean dataset
    profile_clean = profile_clean.merge(event_counts, left_on='id', right_on='person', how='left')

    # Step 4: Drop redundant column (if necessary) and fill NaN values with 0 for the new columns
    profile_clean.drop(columns=['person'], inplace=True)
    profile_clean.fillna(0, inplace=True)

    return prepare_data(profile_clean,portfolio_clean, transcript_clean )


def prepare_data(profile_clean, portfolio_clean,transcript_clean):
    """
        Prepare the cleaned datasets for further modeling by combining user profiles, offers, and user interactions.

        This function takes the cleaned versions of the profile, portfolio, and transcript datasets and:
        - Creates a feature set and corresponding labels for each user-offer combination.
        - Encodes categorical variables such as gender and channels.
        - Simplifies the resulting dataset by dropping unnecessary columns.
        - Removes duplicate rows and handles duplicate labeling to ensure consistency.
        - Merges the portfolio data to enrich the dataset and calculates response rates for different user segments.

        Parameters:
        - profile_clean (pd.DataFrame): Cleaned profile dataset containing user demographic data.
        - portfolio_clean (pd.DataFrame): Cleaned portfolio dataset containing offer details and encoded channel information.
        - transcript_clean (pd.DataFrame): Cleaned transcript dataset containing user interaction events.

        Returns:
        - pd.DataFrame: A merged dataset containing user profiles, offers, and calculated response rates.

        Steps:
        1. Extract relevant columns from the cleaned datasets and rename them for easier merging.
        2. Loop through users and offers to create feature rows, including user demographics, offer properties, and interaction labels.
        3. One-hot encode categorical variables like gender and offer channels, and simplify columns by dropping redundant features.
        4. Drop duplicate feature rows, group them, and update labels based on the most common value within each group.
        5. Merge the enriched portfolio dataset with the final user-offer dataset.
        6. Calculate response rates by age group and gender, providing insights into how different segments respond to offers.

    """
    
    # Step 1: Prepare the dataframes for merging
    profile_df = profile_clean[['id', 'age', 'gender', 'income']].rename(columns={'id': 'user_id'})
    portfolio_df = portfolio_clean[['id', 'reward', 'difficulty', 'duration', 
                                    'channel_email', 'channel_mobile', 'channel_social', 'channel_web']].rename(columns={'id': 'offer_id'})

    # Step 2: Initialize the list to collect data
    data = []

    # Step 3: Loop through each user and each offer to create the feature set and label
    for user_id in profile_df['user_id']:
        # Get user demographics
        user_row = profile_df[profile_df['user_id'] == user_id].iloc[0]
        age, gender, income = user_row['age'], user_row['gender'], user_row['income']
    
        # Get user offer response
        user_response = user_offers_response(user_id,transcript_clean)
    
        for offer_id in portfolio_df['offer_id']:
            # Get offer properties
            offer_row = portfolio_df[portfolio_df['offer_id'] == offer_id].iloc[0]
            reward, difficulty, duration = offer_row['reward'], offer_row['difficulty'], offer_row['duration']
            channel_email = offer_row['channel_email']
            channel_mobile = offer_row['channel_mobile']
            channel_social = offer_row['channel_social']
            channel_web = offer_row['channel_web']
        
            # Determine label for the user-offer combination
            if offer_id in user_response['complete_after_view']:
                label = 1  # complete_after_view
            elif offer_id in user_response['view_without_complete']:
                label = 0  # view_without_complete
            elif offer_id in user_response['complete_without_view']:
                label = 2  # complete_without_view
            else:
                continue  # Skip if there's no relevant interaction

            # Append the row of data to the list
            data.append([
                user_id, offer_id, age, gender, income, reward, difficulty, duration, 
                channel_email, channel_mobile, channel_social, channel_web, label
            ])

    # Step 4: Create a DataFrame from the collected data
    columns = [
        'user_id', 'offer_id', 'age', 'gender', 'income', 'reward', 'difficulty', 'duration', 
        'channel_email', 'channel_mobile', 'channel_social', 'channel_web', 'label'
    ]
    dataset = pd.DataFrame(data, columns=columns)

    # Step 5: Encode categorical variables
    # One-hot encode 'gender' and channels columns
    dataset = pd.get_dummies(dataset, columns=['gender', 'channel_email', 'channel_mobile', 'channel_social', 'channel_web'])

    # Step 6: Drop user_id and offer_id (since they are identifiers, not features)
    dataset.drop(['user_id', 'offer_id'], axis=1, inplace=True)

    # The resulting dataset now has features in `X` and labels in `Y`
    X = dataset.drop(['label'], axis=1)
    y = dataset['label']

    # Simplify channel columns
    dataset['channel_email'] = dataset['channel_email_1']
    dataset['channel_mobile'] = dataset['channel_mobile_1']
    dataset['channel_social'] = dataset['channel_social_1']
    dataset['channel_web'] = dataset['channel_web_1']


    channel_columns = ['channel_email', 'channel_mobile', 'channel_social', 'channel_web']
    dataset[channel_columns] = dataset[channel_columns].astype(int)

    # Drop the original one-hot encoded channel columns
    dataset.drop(['channel_email_1', 'channel_mobile_0', 'channel_mobile_1', 
                  'channel_social_0', 'channel_social_1', 'channel_web_0', 'channel_web_1'], axis=1, inplace=True)


    # Simplify gender column
    def simplify_gender(row):
        if row['gender_F'] == 1:
            return 0
        elif row['gender_M'] == 1:
            return 1
        else:
            return 2

    dataset['gender'] = dataset.apply(simplify_gender, axis=1)

    # Drop the original one-hot encoded gender columns
    dataset.drop(['gender_F', 'gender_M', 'gender_O','channel_email'], axis=1, inplace=True)

    # The modified dataset without label 2
    dataset_wo2 = dataset[(dataset['label'] != 2)]
    # Identifying duplicated rows based on feature columns (excluding the label)
    duplicates = dataset_wo2.loc[dataset_wo2.drop('label', axis=1).duplicated(keep=False)]

    # Grouping duplicates by their features
    grouped_duplicates = duplicates.groupby(list(dataset_wo2.drop('label', axis=1).columns))

    # Creating a dictionary to store new labels for each group
    label_updates = {}

    # Loop through each group of duplicates and set the most common label
    for _, group in grouped_duplicates:
        # Find the most common label in the group
        most_common_label = Counter(group['label']).most_common(1)[0][0]
        # Store the new label
        label_updates.update({idx: most_common_label for idx in group.index})

    # Update the labels in the original dataset
    dataset_wo2.loc[label_updates.keys(), 'label'] = dataset_wo2.loc[label_updates.keys()].index.map(label_updates)
    df_h = dataset_wo2.merge(portfolio_clean, left_on= ['reward','difficulty','duration','channel_social','channel_web'], right_on =['reward','difficulty','duration','channel_social','channel_web'] )
    
    # Calculate response rates
    bins = [17, 24, 34, 44, 54, 64, 74, 100]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
    df_h['age_group'] = pd.cut(df_h['age'], bins=bins, labels=labels)


    response_summary = df_h.groupby(['age_group', 'gender', 'id'], observed = False).agg(
        total_offers=('label', 'count'),
        responses=('label', 'sum'),
    ).reset_index()

    # Calculate response rate
    response_summary['response_rate'] = response_summary['responses'] / response_summary['total_offers']


    response_merged = response_summary.sort_values(['response_rate','total_offers'],ascending = False).merge(portfolio[['id','offer_type','reward','duration','difficulty','channels']], left_on= 'id', right_on= 'id')
    return response_merged






def save_data(response_merged, csv_file_name):
    """
    Save the cleaned data to a CSV.

    Args:
    response_merged: pandas DataFrame. The cleaned data.
    database_filename: str. The filename for the CSV.

    Returns:
    None
    """
    response_merged.to_csv(csv_file_name, index=False)
    
    pass


def main():
    if len(sys.argv) == 5:

        portfolio_filepath, profile_filepath, transcript_filepath, csv_filepath= sys.argv[1:]

        print('Loading data...\n    portfolio: {}\n    profile: {}\n transcript: {}'
              .format(portfolio_filepath, profile_filepath, transcript_filepath))
        df1, df2, df3 = load_data(portfolio_filepath, profile_filepath, transcript_filepath)

        print('Cleaning data...')
        df = clean_data(df1, df2, df3)
        
        print('Saving data...\n    CSV: {}'.format(csv_filepath))
        save_data(df, csv_filepath)
        
        print('Cleaned data saved to the CSV file!')
    
    else:
        print('Please provide the filepaths of the portfolio,  profile , and transcript'\
              'datasets as the first, second, and third argument respectively, as '\
              'well as the filepath of the csv to save the cleaned data '\
              'to as the forth argument. \n\nExample: python data/etl.py '\
              'data/portfolio.json data/profile.json  data/transcript.json '\
              'ata/offer_recommendations.csv')


if __name__ == '__main__':
    main()