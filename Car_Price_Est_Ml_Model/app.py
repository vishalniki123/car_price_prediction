import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

model = pk.load(open('model.pkl','rb'))

st.header('Car Price Prediction ML Model')

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', cars_data['name'].unique())
model_name = st.selectbox('Select Car Model', ['Maruti Swift', 'Skoda Rapid', 'Honda City', 'Hyundai i20',
                                               'Hyundai Xcent', 'Maruti Wagon', 'Maruti 800', 'Toyota Etios',
                                               'Ford Figo', 'Renault Duster', 'Maruti Zen', 'Mahindra KUV',
                                               'Maruti Ertiga', 'Maruti Alto', 'Mahindra Verito', 'Honda WR-V',
                                               'Maruti SX4', 'Tata Tigor', 'Maruti Baleno', 'Chevrolet Enjoy',
                                               'Maruti Omni', 'Maruti Vitara', 'Hyundai Verna', 'Datsun GO',
                                               'Tata Safari', 'Jeep Compass', 'Toyota Fortuner', 'Toyota Innova',
                                               'Mercedes-Benz B', 'Honda Amaze', 'Mitsubishi Pajero',
                                               'Maruti Ciaz', 'Honda Jazz', 'Audi A6', 'Toyota Corolla',
                                               'Mercedes-Benz New', 'Tata Manza', 'Hyundai i10',
                                               'Volkswagen Ameo', 'Volkswagen Vento', 'Ford EcoSport', 'BMW X1',
                                               'Maruti Celerio', 'Volkswagen Polo', 'Maruti Eeco',
                                               'Mahindra Scorpio', 'Ford Freestyle', 'Volkswagen Passat',
                                               'Tata Indica', 'Mahindra XUV500', 'Tata Indigo', 'Nissan Terrano',
                                               'Hyundai Creta', 'Renault KWID', 'Hyundai Santro', 'Audi Q5',
                                               'Lexus ES', 'Jaguar XF', 'Jeep Wrangler', 'Land Rover',
                                               'Mercedes-Benz S-Class', 'BMW 5', 'BMW X4', 'Skoda Superb',
                                               'Mercedes-Benz E-Class', 'MG Hector', 'Volvo XC40', 'Audi Q7',
                                               'Hyundai Elantra', 'Jaguar XE', 'Tata Nexon', 'Mercedes-Benz CLA',
                                               'Toyota Glanza', 'BMW 3', 'Toyota Camry', 'Volvo XC90',
                                               'Maruti Ritz', 'Hyundai Grand', 'Daewoo Matiz', 'Tata Zest',
                                               'Hyundai Getz', 'Hyundai Elite', 'Honda Brio', 'Tata Hexa',
                                               'Nissan Sunny', 'Nissan Micra', 'Mahindra Ssangyong',
                                               'Mahindra Quanto', 'Hyundai Accent', 'Maruti Ignis',
                                               'Mahindra Marazzo', 'Tata Tiago', 'Mahindra Thar', 'Tata Sumo',
                                               'Tata New', 'Mahindra Bolero', 'Mercedes-Benz GL-Class',
                                               'Chevrolet Beat', 'Maruti A-Star', 'Mahindra XUV300', 'Tata Nano',
                                               'Volkswagen GTI', 'Volvo V40', 'Honda CR-V', 'Hyundai EON',
                                               'Datsun RediGO', 'Chevrolet Captiva', 'Ford Fiesta', 'Kia Seltos',
                                               'Honda Civic', 'Chevrolet Sail', 'Tata Venture', 'Ford Classic',
                                               'Honda BR-V', 'Ford Ecosport', 'Tata Aria', 'Mahindra TUV',
                                               'Tata Bolt', 'Honda Accord', 'Mahindra Xylo', 'Fiat Grande',
                                               'Maruti S-Cross', 'Toyota Yaris', 'Chevrolet Tavera', 'Fiat Linea',
                                               'Ford Endeavour', 'Chevrolet Aveo', 'Renault Triber',
                                               'Ford Fusion', 'Skoda Octavia', 'Audi A4', 'Maruti XL6',
                                               'Hyundai Santa', 'Chevrolet Spark', 'Ford Aspire',
                                               'Chevrolet Optra', 'Honda Mobilio', 'Honda BRV', 'BMW X6',
                                               'Chevrolet Cruze', 'Mercedes-Benz GLA', 'BMW 6',
                                               'Mahindra NuvoSport', 'Renault Scala', 'Renault Lodgy',
                                               'Renault Pulse', 'Mahindra Supro', 'Hyundai Sonata',
                                               'Mahindra Renault', 'Nissan Kicks', 'Volkswagen Jetta',
                                               'Mercedes-Benz M-Class', 'Nissan Teana', 'Skoda Yeti', 'Audi Q3',
                                               'Force Gurkha', 'Mahindra Logan', 'Audi A3', 'Maruti Dzire',
                                               'Ford Ikon', 'Renault Fluence', 'Tata Xenon', 'Force One', 'BMW 7',
                                               'Volvo S60', 'Mitsubishi Lancer', 'BMW X7', 'Skoda Fabia',
                                               'Toyota Platinum', 'Renault Captur', 'Maruti Gypsy',
                                               'Renault Koleos', 'Ambassador CLASSIC', 'Tata Harrier',
                                               'Fiat Punto', 'Fiat Avventura', 'Skoda Laura', 'Ashok Leyland',
                                               'Isuzu MUX', 'Opel Astra', 'Hyundai Tucson', 'Maruti Esteem',
                                               'Tata Winger', 'Ambassador Classic', 'Ambassador Grand',
                                               'Toyota Qualis', 'Tata Spacio', 'Hyundai Venue',
                                               'Volkswagen CrossPolo', 'Skoda Kodiaq', 'Isuzu D-Max', 'BMW X3',
                                               'Toyota Land', 'BMW X5', 'Chevrolet Trailblazer', 'Isuzu MU',
                                               'Mercedes-Benz GLC', 'Volvo XC60', 'Volvo S90', 'Maruti S-Presso'])
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller  type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[name,year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
    columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
    
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],
                           [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          list(range(1, 32)), inplace=True)
   

    car_price = model.predict(input_data_model)

    st.markdown('Car Price is going to be '+ str(car_price[0]))
