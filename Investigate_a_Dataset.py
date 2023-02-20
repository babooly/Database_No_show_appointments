#!/usr/bin/env python
# coding: utf-8

# # Investigate a Dataset - [<font color='red'>Medical Appointment No Shows 2016</font>]
# 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment.
# 
# 
# 
# 
# 
# ### Dataset Description 
# 
# * **PatientId:** Identification of a patient.
# 
# * **AppointmentID:** Identification of each appointment.
# 
# * **Gender:** Male or Female.
# 
# * **ScheduledDay:** tells us on what day the patient set up their appointment.
# 
# * **AppointmentDay:** tells us on what day the patient should come for the treatment.
# 
# * **Age:** How old is the patient.
# 
# * **Neighborhood:** indicates the location of the hospital.
# 
# * **Scholarship:** indicates whether or not the patient is enrolled in Brasilian welfare program [Bolsa Família](https://en.wikipedia.org/wiki/Bolsa_Fam%C3%ADlia).
#  
# * **Hipertension:** true or false.
# 
# * **Diabetes:** true or false.
# 
# * **Alcoholism:** true or false.
# 
# * **Handcap:** true or false.
# 
# * **SMS_received:** 1 or more messages sent to the patient.
# 
# * **no show:** it says `‘No’` if the patient showed up to their appointment, and `‘Yes’` if they did not show up.
# 
# ### Question(s) for Analysis
# * **Is there a relationship between age and non-attendance?**
# * **Does gender and age affect attendance?**
# * **Is there a relationship between age, gender, and disease?**
# * **Is there a relationship between sms and non-attendance?**
# * **Does neighbourhood affect attendance?**

# In[219]:


import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# we will load in the data, check for cleanliness, and then trim and clean our dataset for analysis
# ### General Properties

# In[220]:


# Loading the data
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')


# In[221]:


# Displaying the first 5 rows 
df.head()


# ### Assessing

# In[222]:


# Checking to see if any data is missing!
df.info()


# In[223]:


# How many rows and columns does this data contain?
df.shape


# In[224]:


# Examine any missing data rows! 
df.isnull().sum()


# In[225]:


# Checking if there are any duplicated data
df.duplicated().sum()


# It's looks like there are no missing data or duplicated rows.

# In[226]:


# checking the unique values
df.nunique()


# As we can see, there are 62299 unique ids, implying that duplicate ids exist.

# In[227]:


# is there any duplicated ids?
df.PatientId.duplicated().sum()


# It appears that we have duplicates that need to be removed.

# In[228]:


# what is the data types? 
df.dtypes


# it seems like ScheduledDay and AppointmentDay are objects we have to change them to a datetime.

# In[229]:


df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])


# In[230]:


#checking if the types have changed
df['ScheduledDay'].dtype , df['AppointmentDay'].dtype


# In[231]:


# checking the data
df.head()


# In[232]:


# checking the statistics
df.describe()


# As we can see, there is something wrong at the age column there is a minus number, so we have to check that.

# In[233]:


df.Age.value_counts()


# As we can see, there are 3539 rows with age 0, which I think are babies i'm not sure, so we can keep that and simply delete the minus age,in the last row.

# ### Data Cleaning

# In[234]:


# removing the minus age row 
minus_Age = df[df['Age'] == -1]
df.drop(minus_Age.index, inplace=True)


# In[235]:


# removing the duplicated IDs 
df.drop_duplicates('PatientId',inplace=True)


# **Now I am going to rename the data columns and delete the unnecessary columns.**

# In[236]:


# changing the names of Hipertension , No-show and Handcap
df.rename(columns={'Hipertension':'Hypertension','No-show':'No_show','Handcap':'Handicap'},inplace=True)


# In[237]:


# removing unnecessary columns
df.drop(['PatientId','AppointmentID'] ,axis=1, inplace=True)


# In[238]:


df.rename(columns=lambda x: x.strip().lower(),inplace=True)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# ### Research Question 1 (Is there a relationship between age and non-attendance?)

# In[239]:


df.head()


# In[240]:


# Checking the statistics of age
df.age.describe()


# In[241]:


bin_names = ['Childhood','Adult','Middle Age Adult','Elderly']
bin_edges = [0,17,36,56,115]
df['age_stages'] = pd.cut(df.age , bin_edges ,labels=bin_names)


# In[242]:


age_stages = df.groupby('age_stages').age.mean()
age_stages


# In[243]:


# Create a bar chart for age
plt.subplots(figsize=(5,3));
plt.bar(age_stages.index , age_stages , color="C02");
plt.title('average age ');


# As we can see, there are a large number of old patients.

# **making a filter for patiants**

# In[244]:


no_attend = df['no_show']=='Yes'
attend = df['no_show']=='No'


# In[245]:


# how many patients attending the appointment?
plt.bar(["no attend",'attend'],[df[no_attend].shape[0],df[attend].shape[0]]);
plt.title('Number of attending and non-attending patients');
plt.ylabel('number of patients');


# In[246]:


df[no_attend].shape[0],df[attend].shape[0]


# As we can see, 12193 patients didn't attend the appointment.

# In[247]:


# a hist for patients didn't attend
df[no_attend]['age'].hist(figsize=(8,5),color="darkred" );
plt.ylabel('number of patients');
plt.xlabel("Age");


# In[248]:


# a hist for patients attended
df[attend]['age'].hist(figsize=(8,5),color="salmon");
plt.ylabel('number of patients');
plt.xlabel("Age");


# In[249]:


# Comparing attendance and non-attendance 
df[no_attend]["age"].plot(kind="hist", alpha= 1 ,color="darkred", label="no attend", grid=True);
df[attend]["age"].plot( kind="hist",alpha = .5,color="salmon", label="attended", grid=True);
plt.legend();
plt.title('Patients age comparison')
plt.xlabel("Age");
plt.ylabel('number of patients');


# In[250]:


df[attend]["age"].value_counts() , df[no_attend]["age"].value_counts() 


# As we can see, a lot of patients attending the appointment.

# ### Research Question 2  (Does gender and age affect attendance?)

# In[251]:


df.head()


# In[252]:


# statistics for gender
df.gender.describe()


# In[253]:


df.gender.value_counts().plot(kind='pie');


# In[254]:


df.gender.value_counts()


# The number of females is greater than the number of males.

# In[255]:


df['gender'][attend].value_counts() , df['gender'][no_attend].value_counts()


# As we can see, there are 7828 females and 4365 males who did not attend.

# In[256]:


df[attend].groupby('gender').age.mean() , df[no_attend].groupby('gender').age.mean()


# As we can see, females are more present than males for appointments according to age mean.

# In[257]:


# Does gender and age affect attendance?
df[attend].groupby('gender').age.mean().plot(kind='bar' , alpha= 1 ,color="salmon", label="attend")
df[no_attend].groupby('gender').age.mean().plot(kind='bar' , alpha= 1 ,color="darkred", label="no attend")
plt.legend();
plt.title('comparing gender and age')
plt.ylabel('mean age');


# There is no significant effect of gender on attendance or non-attendance.

# ### Research Question 3  (Is there a relationship between age, gender, and disease?)

# In[258]:


df.head()


# In[259]:


df.handicap.value_counts() , df.alcoholism.value_counts()


# In[260]:


df.diabetes.value_counts() , df.hypertension.value_counts()


# **As we can see, handicaps are separated into 5 unique values, and there is not much detail about each one, so I will change that to true and false, and as we can see there are a large number of patients suffering from hypertension and diabetes.**

# In[261]:


for X in [1,2,3,4]:
    df['handicap'].replace(X,1,inplace=True)


# In[262]:


# checking the values
df.handicap.value_counts()


# In[263]:


df[attend].groupby('handicap').age.mean() , df[no_attend].groupby('handicap').age.mean()


# In[264]:


# Is attendance affected by handicap and age? 
df[attend].groupby('handicap').age.mean().plot(kind='bar' , alpha= 1 ,color="salmon", label="attend")
df[no_attend].groupby('handicap').age.mean().plot(kind='bar' , alpha= 1 ,color="darkred", label="no attend")
plt.legend();
plt.title('comparing handicaps and age')
plt.xlabel('handicaps')
plt.ylabel('Mean age');


# In[265]:


# Is attendance affected by alcoholism and age? 
df[attend].groupby('alcoholism').age.mean().plot(kind='bar' , alpha= 1 ,color="salmon", label="attend")
df[no_attend].groupby('alcoholism').age.mean().plot(kind='bar' , alpha= 1 ,color="darkred", label="no attend")
plt.legend();
plt.title('comparing alcoholism and age')
plt.xlabel('alcoholism')
plt.ylabel('Mean age');


# In[266]:


# Does handicap and alcoholism according to age have an impact?
df[attend].groupby(['handicap','alcoholism']).age.mean().plot(kind='bar',alpha= 1 ,color="salmon", label="attend")
df[no_attend].groupby(['handicap','alcoholism']).age.mean().plot(kind='bar',alpha= 1 ,color="darkred", label="no attend")
plt.legend()
plt.title('comparing handicaps and alcoholism according to age')
plt.ylabel('Mean age');


# In[267]:


# Does hypertension and diabetes according to age have an impact?
df[attend].groupby(['hypertension','diabetes']).age.mean().plot(kind='bar',alpha= 1 ,color="salmon", label="attend")
df[no_attend].groupby(['hypertension','diabetes']).age.mean().plot(kind='bar',alpha= 1 ,color="darkred", label="no attend")
plt.legend()
plt.title('comparing hypertension and diabetes according to age')
plt.ylabel('Mean age');


# In[268]:


# Does handicap and alcoholism affect attendance based on gender and age? 
df[attend].groupby(['gender','handicap','alcoholism']).age.mean().plot(kind='bar',alpha= 1 ,color="salmon", label="attend")
df[no_attend].groupby(['gender','handicap','alcoholism']).age.mean().plot(kind='bar',alpha= 1 ,color="darkred", label="no attend")
plt.legend()
plt.title('handicap and alcoholism compared according to A:G')
plt.ylabel('Mean age');


# In[269]:


# Does hypertension and diabetes affect attendance based on gender and age?
df[attend].groupby(['gender','hypertension','diabetes']).age.mean().plot(kind='bar',alpha= 1 ,color="salmon", label="attend")
df[no_attend].groupby(['gender','hypertension','diabetes']).age.mean().plot(kind='bar',alpha= 1 ,color="darkred", label="no attend")
plt.legend()
plt.title('hypertension and diabetes compared according to A:G')
plt.ylabel('Mean age');


# In[270]:


plt.figure(figsize=(14,6))
df[attend].groupby(['gender','hypertension','diabetes','handicap','alcoholism']).age.mean().plot(kind='bar',alpha= 1 ,color="salmon", label="attend")
df[no_attend].groupby(['gender','hypertension','diabetes','handicap','alcoholism']).age.mean().plot(kind='bar',alpha= 1 ,color="darkred", label="no attend")
plt.legend()
plt.title('')
plt.ylabel('Mean age');


# ### Research Question 4  (Is there a relationship between sms and non-attendance?)

# In[271]:


df.head()


# In[272]:


df.sms_received.value_counts()


# **As we can see, a large number of SMS were not received.**

# In[273]:


df['sms_received'][attend].value_counts()  , df['sms_received'][no_attend].value_counts() 


# **35202 patients attend without receiving SMS, and 14903 attend with SMS.
# 6702 didn't attend without SMS. 5491 didn't attend with SMS.**

# In[274]:


# Does SMS affect attendance?
df['sms_received'][attend].value_counts().plot(kind='bar',alpha= 1 ,color="salmon", label="attend")
df['sms_received'][no_attend].value_counts().plot(kind='bar',alpha= 1 ,color="darkred", label="no attend")
plt.legend();
plt.title('SMS reciving')
plt.xlabel('SMS');
plt.ylabel('number of patients');


# **As we can see, patients who do not receive SMS are more committed to their appointments.**

# ### Research Question 5  (Does neighbourhood affect attendance?)

# In[275]:


df.head()


# In[276]:


df['neighbourhood'].nunique()


# In[277]:


df['neighbourhood'].value_counts()


# In[278]:


df['neighbourhood'][attend].value_counts() , df['neighbourhood'][no_attend].value_counts()


# In[279]:


plt.figure(figsize=(15,3))
df['neighbourhood'][attend].value_counts().plot(kind='bar',alpha= 1 ,color="salmon");


# In[280]:


plt.figure(figsize=(15,3))
df['neighbourhood'][no_attend].value_counts().plot(kind='bar',alpha= 1 ,color="darkred");


# In[281]:


# Does the neighbourhoods affect attendance?
plt.figure(figsize=(15,6))
df['neighbourhood'][attend].value_counts().plot(kind='bar',alpha= 1 ,color="salmon" ,label='attend');
df['neighbourhood'][no_attend].value_counts().plot(kind='bar',alpha= 1 ,color="darkred",label='no attend');
plt.legend()
plt.xlabel('neighbourhood')
plt.ylabel('Number of patients');


# **As we can see, neighbourhoods have an impact on attendance.**

# <a id='conclusions'></a>
# ## Conclusions
# The number of patients who don't receive SMS and attend the appointment outnumbers the number of patients who receive SMS but don't attend so i think we need to get much information about SMS and Neighborhoods have an impact on attendance.
# 
# ## limitations
# There is no relationship between gender and attending the appointment.

# In[282]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

