import pandas as pd

phone_usage_cleaned = pd.read_json(path_or_buf="phone_usage3.json")

phone_usage_cleaned = phone_usage_cleaned[["header", "time"]]
phone_usage_cleaned = phone_usage_cleaned.rename(columns={'header': 'Appname',
                            'time': 'Timestamp'})
phone_usage_cleaned = phone_usage_cleaned.iloc[::-1].reset_index(drop=True)
phone_usage_cleaned['Timestamp'] = pd.to_datetime(phone_usage_cleaned['Timestamp'])
phone_usage_cleaned['Timestamp'] = phone_usage_cleaned['Timestamp'].dt.strftime("%d/%m/%Y %X")

phone_usage_cleaned.to_csv(path_or_buf="phone_usage_cleaned3.csv")

test = phone_usage_cleaned.Appname.drop_duplicates().reset_index(drop=True)
# print(test[220:240])
# print(phone_usage_cleaned.head())