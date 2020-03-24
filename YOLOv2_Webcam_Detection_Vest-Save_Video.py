#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import time
from darkflow.net.build import TFNet


# In[2]:


options= {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 27000,
    'threshold' : 0.01,
    'gpu': 0.7
}


# In[3]:


tfnet= TFNet(options)


# In[4]:


colors = [tuple(255 * np.random.rand(3)) for i in range(1)]
colors


# In[5]:


capture= cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


# In[6]:


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
            frame= cv2.rectangle(frame, tl, br, color, 2)
            frame= cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
            out.write(frame)
            cv2.imshow('frame',frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    else:
        break


# In[7]:


capture.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




