import random
import numpy as np
import pandas as pd

def sums_randomization(data_table, mut_groups, features, randomization_number=10000):
    """
    Perform feature-wise randomization test for given mutation groups.

    Parameters:
    data_table (pd.DataFrame): The main data table containing patient data.
    mut_groups (dict): A dictionary where keys are group names and values are lists of 'Patient_ID's for each group.
    features (list): List of feature column names to analyze.
    randomization_number (int): The number of randomizations to perform (default is 10,000).

    Returns:
    pd.DataFrame: Table of feature-wise sum comparison probabilities.
    pd.DataFrame: Table of true feature sums for each group.
    """
    
    # Collect all patients from all groups into one cohort
    whole_cohort = []
    for group in mut_groups:
        whole_cohort += mut_groups[group]
    
    sums_counter_list = []
    true_means_list = []
    
    # Iterate over each mutation group for randomization
    for mut_group in mut_groups:
        # Extract the true data for the current group
        true_data = data_table.loc[data_table.Patient_ID.isin(mut_groups[mut_group]), features]
        
        # List to store randomized samples
        samples = []
        sample_size = len(mut_groups[mut_group])
        
        # Generate randomized cohorts
        while len(samples) < randomization_number:
            new_rand = set(random.sample(list(whole_cohort), k=sample_size))  # Randomly sample patients
            samples.append(new_rand)
        
        # Initialize counters for sums comparison
        sums_counter_list_feature = np.zeros(len(features))
        true_sums = true_data.sum(axis=0)
        
        # Compare true sums against randomized samples
        for sample in samples:
            sample = list(sample)
            rand_data = data_table.loc[data_table.Patient_ID.isin(sample), features]
            rand_sums_sample = rand_data.sum(axis=0)
            
            # Feature-wise comparison
            for i, feature in enumerate(features):
                true_sum = true_sums[feature]
                rand_sum = rand_sums_sample[feature]
                if true_sum > rand_sum:
                    sums_counter_list_feature[i] += 1
        
        # Calculate the fraction of times the true sum exceeded the random sum (p-value approximation)
        sums_counter_list_feature /= randomization_number
        
        # Append results for the current group
        sums_counter_list.append(sums_counter_list_feature)
        true_means_list.append(true_sums.tolist())
    
    # Create final DataFrames with sums and p-value-like statistics
    sums_counter_table = pd.DataFrame(data=sums_counter_list, columns=features, index=mut_groups).T
    true_means_df = pd.DataFrame(data=true_means_list, columns=features, index=mut_groups).T
    
    return sums_counter_table, true_means_df


def samples_randomization(subset_1, subset_2, data_table, features, randomization_number=1000, age_mode=False, female_only_mode=False):
    """
    Perform randomization of samples between two subsets and calculate mean feature values.

    Parameters:
    subset_1 (list): List of patient IDs for the first subset of patients.
    subset_2 (list): List of patient IDs for the second subset of patients.
    data_table (pd.DataFrame): DataFrame containing patient data, with columns 'Patient_ID' and the features.
    features (list): List of feature column names to be analyzed.
    randomization_number (int): Number of random samples to generate (default is 1,000).
    age_mode (bool): If True, matches patients by age (default is False).
    female_only_mode (bool): If True, only considers female patients (default is False).
    Returns:
    tuple: Two DataFrames containing the mean feature values for subset_1 and subset_2 across all random samples.
    """
    
    subset_1_means = []
    subset_2_means = []
    
    # Ensure equal sample sizes for fair comparison
    sample_size = min(len(subset_1), len(subset_2))
    
    for _ in range(randomization_number):
        
        if age_mode:
            # Match by age if age_mode is enabled
            subset_1_rand_sample = []
            subset_2_rand_sample = []
            
            if female_only_mode:
                # Filter data to include only female patients if female_only_mode is enabled
                data_table = data_table[data_table.Gender == 'F']
            
            # Merge subset 1 and 2 by 'Age' to ensure age-matched randomization
            merged_by_age_df = pd.merge(data_table[data_table.Patient_ID.isin(subset_1)], 
                                        data_table[data_table.Patient_ID.isin(subset_2)], 
                                        on='Age', suffixes=['_set_1', '_set_2'])
            
            matched_ages = merged_by_age_df.Age.dropna().unique()
            
            for age in matched_ages:
                age_df = merged_by_age_df[merged_by_age_df.Age == age]
                random_number = random.randint(0, len(age_df) - 1)
                set_1_patient = age_df.iloc[random_number].Patient_ID_set_1
                set_2_patient = age_df.iloc[random_number].Patient_ID_set_2
                subset_1_rand_sample.append(set_1_patient)
                subset_2_rand_sample.append(set_2_patient)
        
        else:
            # Simple random selection without age or gender matching
            subset_1_rand_sample = random.choices(subset_1, k=sample_size)
            subset_2_rand_sample = random.choices(subset_2, k=sample_size)
        
        # Calculate mean feature values for each subset
        subset_1_mean = data_table.loc[data_table.Patient_ID.isin(subset_1_rand_sample), features].mean().tolist()
        subset_1_means.append(subset_1_mean)
        
        subset_2_mean = data_table.loc[data_table.Patient_ID.isin(subset_2_rand_sample), features].mean().tolist()
        subset_2_means.append(subset_2_mean)
        
    # Convert the lists to DataFrames for better readability and return them
    subset_1_means_df = pd.DataFrame(data=subset_1_means, columns=features)
    subset_2_means_df = pd.DataFrame(data=subset_2_means, columns=features)
    
    return subset_1_means_df, subset_2_means_df