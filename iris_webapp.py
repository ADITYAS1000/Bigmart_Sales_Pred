# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:05:13 2020

@author: Aditya
"""

import pickle
from pred_func import predict_sales
import streamlit as st


# Fetching the model
pickle_in = open('Best_model.pkl','rb')
model = pickle.load(pickle_in)


def main():
    
    html_temp = '''
    <div style = 'background-color:#7171da; padding:10px'>
    <h2 style =  'color:white; text-align:center;'>BigMart Sales Prediction</h2>
    </div>
    '''
    st.set_page_config(page_title='BigMart Sales Prediction', 
                            # page_icon = favicon, 
                            # layout = 'wide', 
                            initial_sidebar_state = 'auto')
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    item_weight = st.text_input('Item Weight','format: (00.00)',max_chars=5)
    
    item_visibility = st.text_input('Item Visibility','format: (00.00)%',max_chars=5)
    
    item_mrp = st.text_input('Item MRP','format: (000.00)',max_chars=6)
    
    col1, col2 = st.beta_columns(2)
    with col1:
        item_fat_content = st.selectbox('Item Fat Content',options=['Low Fat','Regular'])
    with col2:
        item_type = st.selectbox('Item Type',
                                 options=['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
                                          'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
                                          'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods',
                                          'Others', 'Seafood'],
                                 )
    
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        outlet_identifier = st.selectbox('Outlet Identifier',options=['OUT010', 'OUT013', 'OUT017', 'OUT018',
                                                                      'OUT019', 'OUT027', 'OUT035', 'OUT045',
                                                                      'OUT046', 'OUT049'])
    with col2:
        outlet_establishment_year = st.selectbox('Outlet_Establishment_Year',
                                 options=[1985, 1987, 1997, 1998, 1999, 2002, 2004, 2007, 2009])
    with col3:
        outlet_size = st.selectbox('Outlet_Establishment_Year',
                                 options=['Small','Medium', 'High'])
    
    col1, col2 = st.beta_columns(2)
    with col1:
        outlet_location_type = st.selectbox('Outlet_Location_Type',
                                 options=['Tier 1', 'Tier 2', 'Tier 3'])
    with col2:
        outlet_type = st.selectbox('Outlet_Establishment_Year',
                                 options=['Grocery Store','Supermarket Type1', 'Supermarket Type2', 
                                          'Supermarket Type3'])
    result = ''
    input_dict = {}
    if st.button('Predict'):
        
        input_dict = {'Item_Weight' : item_weight, 'Item_Fat_Content' : item_fat_content, 
                     'Item_Visibility' : (float(item_visibility)/100.0), 'Item_Type' : item_type, 
                     'Item_MRP' : item_mrp,'Outlet_Identifier': outlet_identifier, 
                     'Outlet_Establishment_Year' : outlet_establishment_year, 'Outlet_Size' : outlet_size, 
                     'Outlet_Location_Type' : outlet_location_type, 'Outlet_Type' : outlet_type}
        
        with open('input_df.pkl', 'wb') as handle:
            pickle.dump(input_dict,handle)
            handle.close()
        
        with open('Best_model.pkl', 'rb') as handle:
            model = pickle.load(handle)
            handle.close()
        
        # with open('build_features.pkl', 'rb') as handle:
        #     features = pickle.load(handle)
        #     handle.close()
            
        result = predict_sales(input_dict,model)
        st.success('The sales for this product is {}'.format(result))
        
if __name__ == '__main__':
    main()
