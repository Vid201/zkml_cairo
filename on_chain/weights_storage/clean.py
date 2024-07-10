import os

data_path = os.path.join('input.data')
data = open(data_path, 'r').read()

cleaned_string = data.replace("Program Output : ", "").strip()

numbers = list(map(int, cleaned_string.strip('[]').split()))

numbers[numbers[0]+1] = int(numbers[numbers[0]+1] / 2)

data_path = os.path.join('input.data')
with open(data_path, "w") as text_file:
    text_file.write(str(numbers).replace(",", ""))
