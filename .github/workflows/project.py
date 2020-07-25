# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
#1. Introduction
#1.1 Description of the Problem
#The population of London has grown considerably over the last decades. London is very diverse. It represents what is called the reflection of the old British Empire. In London, you can get fresh food supplies from Africa. One begins to wonder how efficient the supply mechanism is.

#The real deal is that as much as there are many fine restaurants in London – Asian, Middle Eastern, Latin and American restaurants, you can struggle to find good place to dine in the finest of West African cuisine that has combination of Nigerian, Ghanaian, Cameroonian, Senegalese and more.

#Eating in a cosy environment with a blend of multicultural background and finely made West African dishes, on time and on point in a London location accessible to tourists, within central London and not far from the "unofficial" capital african market place - Peckham.

#1.2 Discussion of the Background
#My client, a successful restaurant chain in Africa is looking to expand operation into Europe through London. They want to create a high-end restaurant that comes with organic mix and healthy. Their target is not only West Africans, but they are pro-organic and healthy eating. To them every meal counts and counts as a royal when you eat.

#Since the London demography is so big, my client needs deeper insight from available data in other to decide where to establish the first Europe “palace” restaurant. This company spends a lot on research and provides customers with data insight into the ingredients used at restaurants.

#1.3 Target Audience
#Considering the diversity of London, there is a high multicultural sense. London is a place where different shades live. As such, in the search for an high-end African-inclined restaurant, there is a high shortage. The target audience is broad, it ranges from Londoners, tourists and those who are passionate about organic food.


#2. Data
#2.1 Description of Data
#This project will rely on public data from Wikipedia and Foursquare.

#2.1.1 Dataset 1:
#In this project, London will be used as synonymous to the "Greater London Area" in this project. Within the Greater London Area, there are areas that are within the London Area Postcode. The focus of this project will be the nieghbourhoods are that are within the London Post Code area.

#The London Area consists of 32 Boroughs and the "City of London". Our data will be from the link - Greater London Area <https://en.wikipedia.org/wiki/List_of_areas_of_London >

#The web scrapped of the Wikipedia page for the Greater London Area data is provided below:


# %%

# library for BeautifulSoup
from bs4 import BeautifulSoup

# library to handle data in a vectorized manner
import numpy as np

# library for data analsysis
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# library to handle JSON files
import json
print('numpy, pandas, ..., imported...')

get_ipython().system('pip -q install geopy')
# conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
print('geopy installed...')
# convert an address into latitude and longitude values
from geopy.geocoders import Nominatim
print('Nominatim imported...')

# library to handle requests
import requests
print('requests imported...')

# tranform JSON file into a pandas dataframe
from pandas.io.json import json_normalize
print('json_normalize imported...')

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
print('matplotlib imported...')

# import k-means from clustering stage
from sklearn.cluster import KMeans
print('Kmeans imported...')

# install the Geocoder
get_ipython().system('pip -q install geocoder')
import geocoder

# import time
import time

# !conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
get_ipython().system('pip -q install folium')
print('folium installed...')
import folium # map rendering library
print('folium imported...')
print('...Done')


# %%
wikipedia_link = 'https://en.wikipedia.org/wiki/List_of_areas_of_London'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:64.0) Gecko/20100101 Firefox/64.0'}
wikipedia_page = requests.get(wikipedia_link, headers = headers)
wikipedia_page


# %%
# Cleans html file
soup = BeautifulSoup(wikipedia_page.content, 'html.parser')
# This extracts the "tbody" within the table where class is "wikitable sortable"
table = soup.find('table', {'class':'wikitable sortable'}).tbody


# %%
# Extracts all "tr" (table rows) within the table above
rows = table.find_all('tr')


# %%
# Extracts the column headers, removes and replaces possible '\n' with space for the "th" tag
columns = [i.text.replace('\n', '')
           for i in rows[0].find_all('th')]


# %%
# Converts columns to pd dataframe
df = pd.DataFrame(columns = columns)
df


# %%
for i in range(1, len(rows)):
    tds = rows[i].find_all('td')
    
    
    if len(tds) == 7:
        values = [tds[0].text, tds[1].text, tds[2].text.replace('\n', ''.replace('\xa0','')), tds[3].text, tds[4].text.replace('\n', ''.replace('\xa0','')), tds[5].text.replace('\n', ''.replace('\xa0','')), tds[6].text.replace('\n', ''.replace('\xa0',''))]
    else:
        values = [td.text.replace('\n', '').replace('\xa0','') for td in tds]
        
        df = df.append(pd.Series(values, index = columns), ignore_index = True)

        df


# %%
df.head(5)


# %%
df = df.rename(index=str, columns = {'Location': 'Location', 'London\xa0borough': 'Borough', 'Post town': 'Post-town', 'Postcode\xa0district': 'Postcode', 'Dial\xa0code': 'Dial-code', 'OS grid ref': 'OSGridRef'})


# %%
df.head(5)


# %%
df['Borough'] = df['Borough'].map(lambda x: x.rstrip(']').rstrip('0123456789').rstrip('['))


# %%
df.shape


# %%
df.head(5)


# %%
df0 = df.drop('Postcode', axis=1).join(df['Postcode'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('Postcode'))


# %%
df0.shape


# %%
df1 = df0[['Location', 'Borough', 'Postcode', 'Post-town']].reset_index(drop=True)


# %%
df1.head(5)


# %%
df1.shape


# %%
df2 = df1
df21 = df2[df2['Post-town'].str.contains('LONDON')]


# %%
df21.head(5)


# %%
df21.shape


# %%
df3 = df21[['Location', 'Borough', 'Postcode']].reset_index(drop=True)


# %%
df3.head(10)


# %%
df_london = df3
df_london.to_csv('LondonLocations.csv', index = False)


# %%
df_london.head(5)


# %%
df_london.Postcode = df_london.Postcode.str.strip()


# %%
df_london.head(5)


# %%
df_se = df_london[df_london['Postcode'].str.startswith(('SE'))].reset_index(drop=True)


# %%
df_se.head(10)


# %%
demograph_link = 'https://en.wikipedia.org/wiki/Demography_of_London'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:64.0) Gecko/20100101 Firefox/64.0'}
demograph_page = requests.get(demograph_link, headers = headers)
soup1 = BeautifulSoup(demograph_page.content, 'html.parser')
table1 = soup1.find('table', {'class':'wikitable sortable'}).tbody
rows1 = table1.find_all('tr')
columns1 = [i.text.replace('\n', '')
           for i in rows1[0].find_all('th')]


# %%
columns1


# %%
demo_london = pd.DataFrame(columns = columns1)


# %%
demo_london.head(5)


# %%



# %%
a


# %%
demo_london['Black'] = demo_london['Black'].astype('float')


# %%



# %%
demo_london_sorted = demo_london.sort_values(by='Black', ascending = False)


# %%
demo_london_sorted.head(5)


# %%
df_se


# %%
df_se_top = df_se[df_se['Borough'].isin(['Lewisham', 'Southwark', 'Lambeth', 'Hackney', 'Croydon'])].reset_index(drop=True)


# %%
df_se_top.head(5)


# %%
df_se_top.shape


# %%
df_se.shape


# %%
def get_latlng(arcgis_geocoder):
    
    
    lat_lng_coords = None
    
    
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}, London, United Kingdom'.format(arcgis_geocoder))
        lat_lng_coords = g.latlng
    return lat_lng_coords


# %%
sample = get_latlng('SE2')
sample


# %%
gg = geocoder.geocodefarm(sample, method = 'reverse')
gg


# %%
start = time.time()

postal_codes = df_se_top['Postcode']    
coordinates = [get_latlng(postal_code) for postal_code in postal_codes.tolist()]

end = time.time()
print("Time of execution: ", end - start, "seconds")


# %%
df_se_loc = df_se_top


df_se_coordinates = pd.DataFrame(coordinates, columns = ['Latitude', 'Longitude'])
df_se_loc['Latitude'] = df_se_coordinates['Latitude']
df_se_loc['Longitude'] = df_se_coordinates['Longitude']


# %%
df_se_loc.head(5)


# %%
df_se_loc.to_csv('SELondonLocationsCoordinates.csv', index = False)


# %%
df_se_loc.shape


# %%
#Please note that due to privacy, the personal Foursquare Credential has been stored in a .json <fsquarecredential.json> and called appropriately as shown below:


# %%
import json
filename = 'foursquareapidata.json'
with open(filename) as f:
    data = json.load(f)


# %%
CLIENT_ID = data['CLIENT_ID'] # your Foursquare ID
CLIENT_SECRET = data['CLIENT_SECRET'] # your Foursquare Secret
VERSION = data['VERSION'] # Foursquare API version


# %%
#3. Methodology
#3.1 Data Exploration
#3.1.1 Single Neighbourhood
#An initial exploration of a single Neighbourhood within the London area was done to examine the Foursquare workability. The Lewisham Borough postcode SE13 and Location - Lewisham is used for this.


# %%
se_df = df_se_loc.reset_index().drop('index', axis = 1)


# %%
se_df.shape


# %%
se_df


# %%
se_df.loc[se_df['Location'] == 'Lewisham']


# %%
se_df.loc[20, 'Location']


# %%
lewisham_lat = se_df.loc[20, 'Latitude']
lewisham_long = se_df.loc[20, 'Longitude']
lewisham_loc = se_df.loc[20, 'Location']
lewisham_postcode = se_df.loc[20, 'Postcode']

print('The latitude and longitude values of {} with postcode {}, are {}, {}.'.format(lewisham_loc,
                                                                                         lewisham_postcode,
                                                                                         lewisham_lat,
                                                                                         lewisham_long))


# %%
LIMIT = 100 
radius = 2000 
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    lewisham_lat, 
    lewisham_long, 
    radius, 
    LIMIT)


url


# %%
results = requests.get(url).json()
results


# %%
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# %%
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) 
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]


nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)


nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]


# %%
nearby_venues


# %%
nearby_venues_lewisham_unique = nearby_venues['categories'].value_counts().to_frame(name='Count')


# %%
nearby_venues_lewisham_unique.head(5)


# %%
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# %%
def getNearbyVenues(names, latitudes, longitudes, radius=2000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
    
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
       
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# %%
se_venues = getNearbyVenues(names=se_df['Location'],
                                   latitudes=se_df['Latitude'],
                                   longitudes=se_df['Longitude']
                                  )


# %%
se_venues.shape


# %%
len(se_venues)


# %%
se_venues['Neighbourhood'].value_counts()
se_venues.to_csv('se_venues.csv')


# %%
se_venues.head(5)


# %%
se_venues.groupby('Neighbourhood').count()


# %%
print('There are {} uniques categories.'.format(len(se_venues['Venue Category'].unique())))


# %%
se_venue_unique_count = se_venues['Venue Category'].value_counts().to_frame(name='Count')


# %%
se_venue_unique_count.head(5)


# %%
se_venue_unique_count.describe()


# %%
# #3.2 Clustering
# For this section, the neighbourhoods in South East London will be clustered based on the processed data obtained above.

# #3.2.1 Libraries
# To get started, all the necessary libraries have been called in the libraries section above.

# #3.2.2 Map Visualization
# Using the geopy library, the latitude and longitude values of London is obtained.


# %%
address = 'London, United Kingdom'

geolocator = Nominatim(user_agent="ln_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of London are {}, {}.'.format(latitude, longitude))


# %%
map_london = folium.Map(location = [latitude, longitude], zoom_start = 12)
map_london


# %%
for lat, lng, borough, loc in zip(se_df['Latitude'], 
                                  se_df['Longitude'],
                                  se_df['Borough'],
                                  se_df['Location']):
    label = '{} - {}'.format(loc, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_london)  
    
display(map_london)


# %%
type(se_df)


# %%
se_onehot = pd.get_dummies(se_venues[['Venue Category']], prefix = "", prefix_sep = "")


# %%
se_onehot['Neighbourhood'] = se_venues['Neighbourhood']


# %%
fixed_columns = [se_onehot.columns[-1]] + list(se_onehot.columns[:-1])
se_onehot = se_onehot[fixed_columns]


# %%
se_onehot.head(5)


# %%
se_onehot.loc[se_onehot['African Restaurant'] != 0]


# %%
se_onehot.loc[se_onehot['Neighbourhood'] == 'Lewisham']


# %%
se_onehot.to_csv('selondon_onehot.csv', index = False)


# %%
se_onehot.shape


# %%
se_grouped = se_onehot.groupby('Neighbourhood').mean().reset_index()
se_grouped.head()


# %%
print("Before One-hot encoding:", se_df.shape)
print("After One-hot encoding:", se_grouped.shape)


# %%
se_grouped.to_csv('london_grouped.csv', index = False)


# %%
num_top_venues = 10 

for hood in se_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = se_grouped[se_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue', 'freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending = False).reset_index(drop = True).head(num_top_venues))
    print('\n')


# %%
#Creating new dataframe:
#Putting the common venues into pandas dataframe, the following return_most_common_venuesis used to sort the venues in descending order.


# %%
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending = False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# %%
num_top_venues = 10

indicators = ['st', 'nd', 'rd']


columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))


neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighbourhood'] = se_grouped['Neighbourhood']

for ind in np.arange(se_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(se_grouped.iloc[ind, :], num_top_venues)


# %%
neighbourhoods_venues_sorted.head()


# %%
neighbourhoods_venues_sorted.to_csv('neighbourhoods_venues_sorted.csv', index = False)


# %%
se_grouped_clustering = se_grouped.drop('Neighbourhood', 1)


# %%
kclusters = 5

kmeans = KMeans(n_clusters = kclusters, random_state=0).fit(se_grouped_clustering)

kmeans.labels_[0:10]


# %%
kmeans.labels_[0:10]


# %%
neighbourhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# %%
se_merged = se_df


# %%
se_merged.head(3)


# %%
se_merged_latlong = se_merged.join(neighbourhoods_venues_sorted.set_index('Neighbourhood'), on = 'Location')


# %%
se_merged_latlong.head(5)


# %%
se_clusters = se_merged_latlong


# %%
#Please note, that the number of clusters was chosen as 5 for initial run.

#3.2.5 Optimal Number of Clusters for K-mean
#To get the optimal number of clusters to be used for the K-mean, there are a number ways possible for the evaluation. Therefore, in this task, the following are used:

#1. Elbow (Criterion) Method 2. Silhouette Coefficient
#1. Elbow Method

#The elbow method is used to solve the problem of selecting k. Interestingly, the elbow method is not perfect either but it gives significant insight that is perhaps not top optimal but sub-optimal to choosing the optimal number of clusters by fitting the model with a range of values for k.

#The approach for this is to run the k-means clustering for a range of value k and for each value of k, the Sum of the Squared Errors (SSE) is calculated., calculate sum of squared errors (SSE). When this is done, a plot of k and the corresponding SSEs are then made. At the elbow (just like arm), that is where the optimal value of k is. And that will be the number of clusters to be used. The whole idea is to have minimum SSE.


# %%
import matplotlib
import numpy as np


# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


sse = {}
for n_cluster1 in range(2, 10):
    kmeans1 = KMeans(n_clusters = n_cluster1, max_iter = 500).fit(se_grouped_clustering)
    se_grouped_clustering["clusters"] = kmeans1.labels_
    
    
    sse[n_cluster1] = kmeans1.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of Clusters, k")
plt.ylabel("Sum of Squared Error, SSE")

plt.vlines(3, ymin = -2, ymax = 45, colors = 'red')
plt.show()


# %%
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

for n_cluster2 in range(2, 10):
    kmeans2 = KMeans(n_clusters = n_cluster2, random_state = 0).fit(se_grouped_clustering)
    label2 = kmeans2.labels_
    sil_coeff = silhouette_score(se_grouped_clustering, label2, metric = 'euclidean')
    print("Where n_clusters = {}, the Silhouette Coefficient is {}".format(n_cluster2, sil_coeff))


# %%
se_clusters.columns


# %%
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, poi, cluster in zip(se_clusters['Latitude'], se_clusters['Longitude'], se_clusters['Location'], se_clusters['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=20,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)

display(map_clusters)


# %%
se_clusters.loc[se_clusters['Cluster Labels'] == 0, se_clusters.columns[[1] + list(range(5, se_clusters.shape[1]))]]


# %%
se_clusters.loc[se_clusters['Cluster Labels'] == 1, se_clusters.columns[[1] + list(range(5, se_clusters.shape[1]))]]


# %%
se_clusters.loc[se_clusters['Cluster Labels'] == 2, se_clusters.columns[[1] + list(range(5, se_clusters.shape[1]))]]


# %%
se_clusters.loc[se_clusters['Cluster Labels'] == 3, se_clusters.columns[[1] + list(range(5, se_clusters.shape[1]))]]


# %%
se_clusters.loc[se_clusters['Cluster Labels'] == 4, se_clusters.columns[[1] + list(range(5, se_clusters.shape[1]))]]


# %%
#4. Result
#The following are the highlights of the 5 clusters above:

#Pubs, Cafe, Coffee Shops are popular in the South East London.
#As for restaurants, the Italian Restaurants are very popular in the South East London area. Especially in Southwark and Lambeth areas.
#With the Lewisham area being the most condensed area of Africans in the South East Area, it is surprising to see how in the top 10 venues, you can barely see restaurants in the top 5 venues.
#Although, the Clusters have variations, a very visible presence is the predominance of pubs.

#5. Discussion and Conclusion
#It is very important to note that Clusters 2 and 3 are the most viable clusters to create a brand African Restaurant. Their proximity to other amenities and accessibility to station are paramount. These 2 clusters do not have top restaurants that could rival their standards if they are created. And the proximity to resources needed is paramount as Lewisham and Lambeth are not far out from Peckham (under Southwark).

#In conclusion, this project would have had better results if there were more data in terms of crime data within the area, traffic access and allowance of more venues exploration with the Foursquare (limited venues for free calls).

#Also, getting the ratings and feedbacks of the current restaurants within the clusters would have helped in providing more insight into the best location.


# %%


