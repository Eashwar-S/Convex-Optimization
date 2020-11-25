import xlrd
import numpy as np

# Give the location of the file
loc = ("data.xlsx")  # Contains weather data of 15 states of US
data = np.arange(180).reshape(15, 12)
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)


for i in range(1, 13):
    for j in range(1, 16):
        data[j-1, i-1] = sheet.cell_value(j, i)


# np.save("data", data)
data1 = np.load("data.npy")
print(data1)
