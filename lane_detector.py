import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import bezier
from scipy.spatial import distance

def boxes_from_gtlanes(lanes,method=None,box_diag=150):
  '''
  param method: 'bezier' for piece wise
  param box_diag: size of subbozes
  TODO: overlap
  '''
  boxes=[]
  assert method in [None, 'bezier']
  
  for lane in lanes:
    #print('lane',lane)
    poly2d = lane['poly2d']
    assert len(poly2d) == 1, 'poly2d length is not one. parse it.'
    nodes = np.array(lane['poly2d'][0]['vertices']) # col,row

    if method is None:

      cmin,rmin = nodes.min(0)
      cmax,rmax = nodes.max(0)
      boxes.append([cmin,rmin,cmax,rmax])

    elif method == 'bezier':
      nodes = np.asfortranarray([nodes[:,0].tolist(),nodes[:,1].tolist()])
      pts = comp_bezier_pts(nodes)
      try:
        sub_boxes = sub_boxes_from_pts(pts,box_diag=box_diag,method='thick')
      except Exception as e:
        sub_boxes = sub_boxes_from_pts(pts,box_diag=box_diag)
        print(e,'. thick method failed for',lane)
      boxes.extend(sub_boxes)
    
  return(boxes)

def pred_lines_to_boxes(lines,method=None,box_diag=10):
  boxes=[]
  
  for line in lines:
    line = line[0]
    
    if method is None:
      cmax,cmin = max(line[0],line[2]), min(line[0],line[2])
      rmax,rmin = max(line[1],line[3]), min(line[1],line[3])
      boxes.append([cmin,rmin,cmax,rmax])
    
    elif method == 'bezier':
      nodes = np.asfortranarray([[line[0],line[2]],
                                 [line[1],line[3]]]
                               ).astype(float)
      pts = comp_bezier_pts(nodes,degree=2,num=100)
      try:
        sub_boxes = sub_boxes_from_pts(pts,box_diag=box_diag,method='thick')
      except Exception as e:
        sub_boxes = sub_boxes_from_pts(pts,box_diag=box_diag)
        print(e,'. thick method failed for',lane)
      boxes.extend(sub_boxes)
  
  return(boxes)

def boxes_to_bool2d(boxes,r,c):
  '''and method faster'''
  bool2d_list = []
  bool2d = np.zeros((r.size,c.size)).astype(bool)

  for box in boxes:
    cmin,rmin,cmax,rmax = box
    rbool = np.logical_and(rmin < r,r < rmax ) # TODO: <, <=, etc
    cbool = np.logical_and(cmin < c,c < cmax )
    bool2d += np.outer(rbool,cbool)
  
  return(bool2d.astype(bool))

def comp_bezier_pts(nodes,degree=4,num=100):
  curve = bezier.Curve(nodes, degree=degree) # 3 or 5 ?
  s = np.linspace(0,1,num) # TODO: 50-300?
  pts = curve.evaluate_multi(s)
  return(pts)

def rolling_window(a, window, step_size):
  '''
  credit: https://gist.github.com/codehacken/708f19ae746784cef6e68b037af65788
  '''
  shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size, window)
  strides = a.strides + (a.strides[-1] * step_size,)
  rw = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
  return(rw)

def sub_boxes_from_pts(pts,box_diag=150,method=None):
  '''
  param box_diag # 50 - 200?
  '''
  boxes=[]
  if method is None:
    n_boxes = np.round(np.linalg.norm(pts.max(1)-pts.min(1)) / box_diag
                      ).astype(int)
    n_boxes = min(n_boxes,pts.shape[1]) # in case n_boxes > number of pts
    n_boxes = max(n_boxes,1)

    segments = np.array_split(pts.T,n_boxes)

  elif method == 'thick':
    
    pt_dist = distance.euclidean(pts[:,0],pts[:,1]) # 100 pts, want image extent
    window = np.round(box_diag / pt_dist).astype(int) # in pix on image, not pts
    c=rolling_window(pts[0], window=window, step_size=1)
    #assert np.isclose(c.max(),pts[0].max())
    #assert np.isclose(c.min(),pts[0].min())
    r=rolling_window(pts[1], window=window, step_size=1)
    #assert np.isclose(r.max(),pts[1].max())
    #assert np.isclose(r.min(),pts[1].min())
    M,N = c.shape
    segments = np.zeros((M,N,2))
    segments[:,:,0] = c
    segments[:,:,1] = r
    
  else:
    assert False
  
  for segment in segments:
   # print('segment',segment)
    cmin,rmin = segment.min(0).astype(int)
    cmax,rmax = segment.max(0).astype(int)
    boxes.append([cmin,rmin,cmax,rmax])
  return(boxes)

def evaluate(bool_2d_1,bool_2d_2):
  assert bool_2d_1.shape == bool_2d_2.shape
  intersection = np.logical_or(bool_2d_1,bool_2d_2).sum()
  overlap = np.logical_and(bool_2d_1,bool_2d_2).sum()
  iou = overlap / intersection
  return(iou)