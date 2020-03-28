from datetime import datetime
import requests
import pandas as pd
import numpy as np
import pickle
import json

class weather_request:
    
    def __init__(self, lat, long, t_start, n_days, timezone):
        self.lat = lat
        self.long = long
        self.n_days = n_days
        self.daysecond = 3600*24
        self.requests_list = []
        self.json_objs = []
        self.timezone = timezone
        
        if(isinstance(t_start, str)):
            datetime_object = datetime.strptime(t_start, "%Y-%m-%d %H:%M")
            
            self.t_start = int(datetime_object.timestamp())
        else:
            raise TypeError('fill in time format "YYYY-MM-DD HH:MM".')
             
        
        
    def request(self):
        for i in range(self.n_days):
            req = requests.get("https://api.darksky.net/forecast/9ae4fb3bd1f83bc1076d8f367b3d27b4/{},{},{}?units=si"
                                   .format(self.lat, self.long, self.t_start + i*self.daysecond))
            self.requests_list.append(req)
        
    def get_pandas(self):
        if len(self.requests_list) == 0:
            raise Exception("user should do the request() first")
        else:
            
            for js_obj in self.requests_list:
                self.json_objs += js_obj.json()['hourly']['data']
            with open('temp.json', 'w') as outfile:
                json.dump(self.json_objs, outfile)
            jspd = pd.read_json('json/temp.json',orient='columns')
            ## data frame cleaning
            jspd['time'] = pd.to_datetime(jspd['time'], unit='s')
            jspd['time'] = jspd['time'].dt.tz_localize("UTC").dt.tz_convert(self.timezone)
            jspd['time'] = jspd['time'].apply(lambda x:datetime.replace(x,tzinfo=None))
            jspd.set_index(jspd['time'], inplace = True)
            jspd.drop(columns = ['time'], inplace = True)
            return jspd
        return 
