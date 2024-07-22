import pandas as pd
def find_s_algorithm(filename, encoding='utf-8'):
    try:
        data = pd.read_csv(filename, encoding=encoding, on_bad_lines='skip')
        attributes = data.columns[:-1]
        class_label = data.columns[-1]
        hypothesis = ['0'] * len(attributes)
        for index, row in data.iterrows():
            if row[class_label] == 'Yes':
                for i, attribute in enumerate(attributes):
                    if hypothesis[i] == '0':
                        hypothesis[i] = row[attribute]
                    elif hypothesis[i] != row[attribute]:
                        hypothesis[i] = '?'
        print("Final Hypothesis:", hypothesis)
    except UnicodeDecodeError:
        print("UnicodeDecodeError: Please check the encoding of the CSV file.")
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
filename = 'C:/Users/HP/OneDrive/Documents/weather_data.csv'
find_s_algorithm(filename, encoding='latin1')
