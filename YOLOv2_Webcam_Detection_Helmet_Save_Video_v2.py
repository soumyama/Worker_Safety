#!/usr/bin/env python
# coding: utf-8

# In[25]:


import cv2
import numpy as np
import time
from darkflow.net.build import TFNet


# In[26]:


options= {
    'model': 'cfg/tiny-yolo-voc-2c.cfg',
    'load': 2750, #prototype1:1750, 
    'threshold' : 0.30,
    'gpu': 0.7
}


# In[27]:


tfnet= TFNet(options)


# In[28]:


colors = [tuple(255 * np.random.rand(3)) for i in range(1)]
colors


# In[29]:


capture= cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


# In[30]:


time_lst=[]
label_lst=[]

start_time=time.time()
while(capture.isOpened()):
    start= time.time()
    ret, frame= capture.read()
    results= tfnet.return_predict(frame)
    
    if ret==True:
        for color, result in zip(colors, results):
            tl= (result['topleft']['x'], result['topleft']['y'])
            br= (result['bottomright']['x'], result['bottomright']['y'])
            label= result['label']
            confidence= result['confidence']
            text= '{}: {:.0f}%'.format(label, confidence*100)
            frame= cv2.rectangle(frame, tl, br, color, 5)
            frame= cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            time_lst.append(time.time())
            label_lst.append(label)
            out.write(frame)
            cv2.imshow('frame',frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    else:
        break

capture.release()
out.release()
cv2.destroyAllWindows()
end_time=time.time()


# In[31]:


time_list_enhanced=[]
for i in range(0, len(time_lst)):
    time_list_enhanced.append(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time_lst[i])))


# In[36]:


import pandas as pd
df= pd.DataFrame({'label':label_lst, 'timestamp':time_list_enhanced})
df.to_csv('YOLO_helmet_detection_data.csv', encoding='utf-8')


# In[ ]:




