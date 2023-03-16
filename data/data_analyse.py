import pandas as pd
import matplotlib.pyplot as plt

# load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# The relationship between price and floor area
plt.scatter(train['floor_area_sqm'], train['resale_price'])
plt.xlabel('Floor Area (sqm)')
plt.ylabel('Resale Price')
plt.title('Resale Price vs Floor Area')
plt.show()

# The relationship between lease commence date and resale price
plt.scatter(train['lease_commence_date'], train['resale_price'])
plt.xlabel('Lease Commence Date')
plt.ylabel('Resale Price')
plt.title('Resale Price vs Lease Commence Date')
plt.show()





