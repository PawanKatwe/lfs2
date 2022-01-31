from re import X
import pandas as pd
import streamlit as st
import pickle 
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train.csv')

def user_input(company,year,mileage,engine,max_power,km_driven,seats,seller_type,fuel_type,transmission_type):
    company = str(company)
    year = int(year)
    mileage = float(mileage)
    engine = float(engine)
    max_power = float(max_power)
    km_driven = float(km_driven)
    seats = float(seats)
    seller_type = str(seller_type)
    fuel_type = str(fuel_type)
    transmission_type = str(transmission_type)


    input = {
    'company':company,
    'year':year,
    'mileage':mileage,
    'engine':engine,
    'max_power':max_power,
    'km_driven':km_driven,
    'seats':seats,
    'seller_type':seller_type,
    'fuel_type':fuel_type,
    'transmission_type':transmission_type
    }

    global input_df
    input_df = pd.DataFrame(input,index=[0])

    return input_df



st.title('Used car price predicton')

company = st.selectbox('Company',('Maruti', 'Hyundai', 'Honda', 'Mahindra', 'Toyota', 'Tata', 'Ford',
    'Volkswagen', 'Renault', 'Mercedes-Benz', 'BMW', 'Skoda', 'Chevrolet',
    'Audi', 'Nissan', 'Datsun', 'Fiat', 'Jaguar', 'Land', 'Jeep', 'Volvo',
    'Mitsubishi', 'Kia', 'Porsche', 'Mini', 'MG', 'Isuzu', 'Lexus','others'))

year = st.slider('Year',1991,2021)

mileage = st.slider('Mileage',1,150)

engine = st.slider('Engine CC',50,7000)

max_power = st.slider('Max Power bhp',25,650)

km_driven = st.slider('Km Driven',100,1000000,1000)

seats = st.slider('Seats',2,14)

seller_type = st.selectbox('Seller Type',('Individual','Dealer','Trustmark Dealer'))

fuel_type = st.selectbox('Fuel Type',('Diesel','Electric','LPG','Petrol','CNG'))

transmission_type = st.selectbox('Transmission Type',('Manual','Automaic'))

input_df = user_input(company,year,mileage,engine,max_power,km_driven,seats,seller_type,fuel_type,transmission_type)

st.write(input_df)

input_df = pd.get_dummies(data=input_df, columns=['seller_type','fuel_type','transmission_type','company'])

x = pd.DataFrame()

x[[ 'year', 'km_driven', 'mileage', 'engine', 'max_power',
       'seats', 'seller_type_Dealer', 'seller_type_Individual',
       'seller_type_Trustmark Dealer', 'fuel_type_CNG', 'fuel_type_Diesel',
       'fuel_type_Electric', 'fuel_type_LPG', 'fuel_type_Petrol',
       'transmission_type_Automatic', 'transmission_type_Manual',
       'company_Audi', 'company_BMW', 'company_Chevrolet', 'company_Datsun',
       'company_Fiat', 'company_Ford', 'company_Honda', 'company_Hyundai',
       'company_Isuzu', 'company_Jaguar', 'company_Jeep', 'company_Kia',
       'company_Land', 'company_Lexus', 'company_MG', 'company_Mahindra',
       'company_Maruti', 'company_Mercedes-Benz', 'company_Mini',
       'company_Mitsubishi', 'company_Nissan', 'company_Porsche',
       'company_Renault', 'company_Skoda', 'company_Tata', 'company_Toyota',
       'company_Volkswagen', 'company_Volvo', 'company_others']] = 0

missing_cols = set(x.columns) - set(input_df.columns)

# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    input_df[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
input_df = input_df[x.columns]

scaler = StandardScaler()
scaler = scaler.fit(train)
df = scaler.transform(input_df)


if st.button('Predict'):
    model_car_price = open('used_car_price_model.pkl','rb')   
    model = pickle.load(model_car_price)
    pred_price = model.predict(df)
    
    st.write('Predicted price of the car is ',pred_price)


