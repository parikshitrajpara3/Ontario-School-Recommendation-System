import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title('School Recommendation System')
st.write('Find similar schools based on performance metrics and location')

@st.cache_data
def load_data():
    # Load your data files
    school = pd.read_excel("general school information (fact).xlsx")
    teacher = pd.read_excel("teachers data.xlsx")
    expulsions = pd.read_excel("Expulsions.xlsx")
    suspension = pd.read_excel("Suspensions.xlsx")
    genders = pd.read_excel("Gender Enrollement.xlsx")
    
    # Define columns to sum
    columns_to_sum = [
        'Percentage of Grade 3 Students Achieving the Provincial Standard in Reading',
        'Percentage of Grade 3 Students Achieving the Provincial Standard in Writing',
        'Percentage of Grade 3 Students Achieving the Provincial Standard in Mathematics',
        'Percentage of Grade 6 Students Achieving the Provincial Standard in Reading',
        'Percentage of Grade 6 Students Achieving the Provincial Standard in Writing',
        'Percentage of Grade 6 Students Achieving the Provincial Standard in Mathematics',
        'Percentage of Grade 9 Students Achieving the Provincial Standard in Academic Mathematics',
        'Percentage of Grade 9 Students Achieving the Provincial Standard in Applied Mathematics',
        'Percentage of Students That Passed the Grade 10 OSSLT on Their First Attempt'
    ]

    columns_to_sum2 = [
        'Percentage of Students Who Are New to Canada from a Non-English Speaking Country',
        'Percentage of Students Who Are New to Canada from a Non-French Speaking Country'
    ]

    # Process percentage columns
    for col in columns_to_sum:
        if col in school.columns:
            school[col] = pd.to_numeric(school[col].str.rstrip('%'), errors='coerce') / 100

    school['overall well performing kids%'] = school[columns_to_sum].mean(axis=1) * 100

    for col in columns_to_sum2:
        if col in school.columns:
            school[col] = pd.to_numeric(school[col].str.rstrip('%'), errors='coerce') / 100

    school['international kids%'] = school[columns_to_sum2].mean(axis=1) * 100
    
    # Process gifted kids percentage
    school['gifted kids%'] = pd.to_numeric(school['Percentage of Students Identified as Gifted'].str.rstrip('%'), errors='coerce')

    # Merge all dataframes
    df1 = pd.merge(school, teacher, on='Board Name', how='outer')
    df2 = pd.merge(df1, expulsions, on=['Board Name', 'Board Number'], how='outer')
    df3 = pd.merge(df2, suspension, on=['Board Name', 'Board Number'], how='outer')
    df4 = pd.merge(df3, genders, on=['Board Number', 'Board Name'], how='outer')
    
    df = df4.copy()
    
    # Clean data
    df = df.dropna(subset=['School Name', 'Total Male', 'Latitude'])
    
    # Fill NA values for specific columns
    col = ['overall well performing kids%', 'international kids%', 'gifted kids%']
    df[col] = df[col].fillna(0)
    
    # Select numeric columns for PCA
    numeric_columns = [
        'overall well performing kids%', 'international kids%', 'gifted kids%',
        'Expulsion Rate', 'Suspension Rate', 'Elementary Male', 'Elementary Female',
        'Secondary Male', 'Secondary Female', 'Total Male', 'Total Female'
    ]
    
    # Prepare data for PCA
    X = df[numeric_columns]
    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=6)  # Using 6 components as in your original code
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=[f'PC{i+1}' for i in range(6)]
    )
    
    # Combine with original dataframe
    final_df = pd.concat([
        df[['School Name', 'Latitude', 'Longitude', 'School Level']],
        pca_df
    ], axis=1)
    
    return final_df.dropna()

def recommend_schools(input_school_name, df, num_recommendations=5, max_distance=float('inf')):
    # Your existing recommend_schools function (unchanged)
    input_school = df[df['School Name'] == input_school_name]
    
    if input_school.empty:
        return "School not found."

    input_school_level = input_school['School Level'].values[0]
    input_lat, input_lon = input_school[['Latitude', 'Longitude']].values[0]
    
    pca_columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']
    input_features = pd.DataFrame(
        input_school[pca_columns + ['Latitude', 'Longitude']].values,
        columns=pca_columns + ['Latitude', 'Longitude']
    )

    filtered_df = df[df['School Level'] == input_school_level]
    
    knn = NearestNeighbors(n_neighbors=num_recommendations + 1, metric='euclidean')
    X_fit = filtered_df[pca_columns + ['Latitude', 'Longitude']]
    knn.fit(X_fit)

    distances, indices = knn.kneighbors(input_features)
    
    recommended_schools = []
    for dist, idx in zip(distances[0][1:], indices[0][1:]):
        recommended_school = filtered_df.iloc[idx]
        school_name = recommended_school['School Name']
        school_lat, school_lon = recommended_school[['Latitude', 'Longitude']]
        
        geo_distance = geodesic((input_lat, input_lon), (school_lat, school_lon)).kilometers
        
        if geo_distance <= max_distance:
            recommended_schools.append((school_name, geo_distance))

    return recommended_schools[:num_recommendations]

# Load the data
try:
    df = load_data()
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Create sidebar for filters
st.sidebar.header('Filters')

# Distance filter
max_distance = st.sidebar.slider('Maximum Distance (km)', 
                               min_value=1, 
                               max_value=100, 
                               value=20)

# Number of recommendations filter
num_recommendations = st.sidebar.slider('Number of Recommendations', 
                                      min_value=1, 
                                      max_value=10, 
                                      value=5)

# School selection
all_schools = sorted(df['School Name'].unique())
selected_school = st.selectbox('Select a School:', all_schools)

# Add a button to trigger recommendations
if st.button('Get Recommendations'):
    recommendations = recommend_schools(
        selected_school, 
        df, 
        num_recommendations=num_recommendations,
        max_distance=max_distance
    )
    
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.success(f"Here are the top {len(recommendations)} recommended schools similar to {selected_school}:")
        
        results_df = pd.DataFrame(recommendations, columns=['School Name', 'Distance (km)'])
        results_df['Distance (km)'] = results_df['Distance (km)'].round(2)
        
        
        results_df.index = results_df.index + 1

        st.dataframe(results_df)
        
        # Display on a map
        st.subheader('School Locations')
        map_data = []
        
        # Add selected school
        selected_school_data = df[df['School Name'] == selected_school]
        map_data.append({
            'lat': selected_school_data['Latitude'].iloc[0],
            'lon': selected_school_data['Longitude'].iloc[0],
            'name': selected_school,
            'type': 'Selected School'
        })
        
        # Add recommended schools
        for school_name, _ in recommendations:
            school_data = df[df['School Name'] == school_name]
            map_data.append({
                'lat': school_data['Latitude'].iloc[0],
                'lon': school_data['Longitude'].iloc[0],
                'name': school_name,
                'type': 'Recommended School'
            })
        
        map_df = pd.DataFrame(map_data)
        st.map(map_df)

