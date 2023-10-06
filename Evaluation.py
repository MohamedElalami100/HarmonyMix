def calculate_precision(test_data, model):
    print(12)
    # Initialize variables to keep track of total precision and the number of users
    total_precision = 0
    num_users = 0

    # Iterate through each user in the test data
    unique_users = test_data['user_id'].unique()

    for user in unique_users:
        # Get the actual songs listened to by the user in the test data
        actual_songs = test_data[test_data['user_id'] == user]['song'].tolist()

        # Get the recommendations from the model for the same user
        recommended_songs = model.recommend(user)
        print(recommended_songs)

        # Calculate the intersection of actual and recommended songs
        intersection = set(actual_songs).intersection(recommended_songs)

        # Calculate precision for this user and add it to the total
        if len(recommended_songs) > 0:
            precision = len(intersection) / len(recommended_songs)
            total_precision += precision
            num_users += 1

    # Calculate the average precision across all users
    if num_users > 0:
        average_precision = total_precision / num_users
        return average_precision
    else:
        return 0



