#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os

import copy

from collections.abc import Iterable
from typing import Union

from dataclasses import dataclass

@dataclass
class PageElement:

    eps: float = 1.e-3
    
    cid: int = -1# conversion id
    pid: int = -1# page-id
    
    x0: float = -1.0# lower-left x
    y0: float = -1.0# lower-left y

    x1: float = -1.0 # upper-right x
    y1: float = -1.0 # upper-right y

    label: str = "<undefined>" # layout label

    def __lt__(self, other):
        if self.pid==other.pid:

            if self.overlaps_x(other):
                return self.y0 > other.y0
            else:
                return self.x0 < other.x0
        else:
            return self.pid<other.pid
        
    def follows_maintext_order(self, rhs) -> bool:
        return (self.cid+1==rhs.cid)

    def overlaps(self, rhs) -> bool:
        return (self.overlaps_x(rhs) and self.overlaps_y(rhs))
    
    def overlaps_x(self, rhs) -> bool:
        return ((self.x0<=rhs.x0 and rhs.x0<self.x1) or
	        (self.x0<=rhs.x1 and rhs.x1<self.x1) or
	        (rhs.x0<=self.x0 and self.x0<rhs.x1) or
	        (rhs.x0<=self.x1 and self.x1<rhs.x1) );
    
    def overlaps_y(self, rhs) -> bool:
        return ((self.y0<=rhs.y0 and rhs.y0<self.y1) or
	        (self.y0<=rhs.y1 and rhs.y1<self.y1) or
	        (rhs.y0<=self.y0 and self.y0<rhs.y1) or
	        (rhs.y0<=self.y1 and self.y1<rhs.y1) );

    def overlaps_y_with_iou(self, rhs, iou:float) -> bool:
        if self.overlaps_y(rhs):
            
            u0 = min(self.y0, rhs.y0);
            u1 = max(self.y1, rhs.y1);
            
            i0 = max(self.y0, rhs.y0);
            i1 = min(self.y1, rhs.y1);
            
            iou_ = float(i1-i0)/float(u1-u0);
            return (iou_)>iou;

        return False;

    def is_left_of(self, rhs) -> bool:
        return (self.x0<rhs.x0)
    
    def is_strictly_left_of(self, rhs) -> bool:
        return ((self.x1+self.eps)<rhs.x0)

    def is_above(self, rhs) -> bool:
        return (self.y0>rhs.y0)
    
    def is_strictly_above(self, rhs) -> bool:
        return ((self.y0+self.eps)>rhs.y1)

    def is_horizontally_connected(self, elem_i, elem_j) -> bool:
        min_ij:float = min(elem_i.y0, elem_j.y0)
        max_ij:float = max(elem_i.y1, elem_j.y1)
        
        if self.y0<max_ij and self.y1>min_ij: # overlap_y
            return False
        
        if self.x0<elem_i.x1 and self.x1>elem_j.x0:
            return True
        
        return False        

class ReadingOrderPredictor:
    r"""
    Rule based reading order for DoclingDocument
    """

    def __init__(self):
        return

    def predict_page(self, page_elems: list[PageElement]) -> list[PageElement]:
        r"""
        Reorder the output of the 
        """
        #doc_elems = self._to_page_elements(conv_res)

        h2i_map, i2h_map = self._init_h2i_map(page_elems)

        l2r_map, r2l_map = self._init_l2r_map(page_elems)

        up_map, dn_map = self._init_ud_maps(page_elems, l2r_map, r2l_map)

        if True:
            dilated_page_elems = copy.deepcopy(page_elems) # deep-copy
            self._do_horizontal_dilation(page_elems, dilated_page_elems, up_map, dn_map);
      
            # redo with dilated provs
            up_map, dn_map = self._init_ud_maps(dilated_page_elems, l2r_map, r2l_map)
            
        heads = self._find_heads(page_elems, h2i_map, i2h_map, up_map, dn_map)
            
        self._sort_ud_maps(page_elems, h2i_map, i2h_map, up_map, dn_map);
        order = self._find_order(page_elems, heads, up_map, dn_map);
        
        sorted_page_elems: list[PageElement] = [];
        for ind in order:
            sorted_page_elems.append(page_elems[ind]);

        return sorted_page_elems

    """
    def _to_page_elements(self, conv_res:ConversionResult):

        self.page_elements = {}
        self.page_elements = {p.page_no: [] for p in conv_res.pages}

        for elem_id, element in enumerate(conv_res.assembled.elements):
            # Convert bboxes to lower-left origin.
            bbox = DsBoundingBox(
                element.cluster.bbox.to_bottom_left_origin(
                    page_no_to_page[element.page_no].size.height
                ).as_tuple()
            )

            elem = PageElement(cid=cid, pid=element.page_no,
                               x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                               label=element.label)
            
            self.page_elements[element.page_no].append(elem)
    """
    
    def _init_h2i_map(self, page_elems: list[PageElement]):            
        h2i_map = {}
        i2h_map = {} 

        for i,pelem in enumerate(page_elems):
            h2i_map[pelem.cid] = i
            i2h_map[i] = pelem.cid
        
        return h2i_map, i2h_map
        
    def _init_l2r_map(self, page_elems: list[PageElement]):
        l2r_map = {}
        r2l_map = {} 

        for i,pelem_i in enumerate(page_elems):
            for j,pelem_j in enumerate(page_elems):        

                if(pelem_i.follows_maintext_order(pelem_j) and
                   pelem_i.is_strictly_left_of(pelem_j) and
                   pelem_i.overlaps_y_with_iou(pelem_j, 0.8)):
                    l2r_map[i] = j;
                    r2l_map[j] = i;
                
        return l2r_map, r2l_map
    
    def _init_ud_maps(self, page_elems: list[PageElement],
                      l2r_map: dict[int, int],
                      r2l_map: dict[int, int]):
        up_map: dict[int, list[int]] = {}
        dn_map: dict[int, list[int]] = {}

        for i,pelem_i in enumerate(page_elems):
            up_map[i] = []
            dn_map[i] = []

        for j,pelem_j in enumerate(page_elems):

            if j in r2l_map:
                i = r2l_map[j]

                dn_map[i] = [j]
                up_map[j] = [i]

                continue

            for i,pelem_i in enumerate(page_elems):

                if i==j:
                    continue

                is_horizontally_connected:bool = False;
                is_i_just_above_j:bool = (pelem_i.overlaps_x(pelem_j) and pelem_i.is_strictly_above(pelem_j));

                for w,pelem_w in enumerate(page_elems):

                    if(not is_horizontally_connected):
                        is_horizontally_connected = pelem_w.is_horizontally_connected(pelem_i, pelem_j);

                    # ensure there is no other element that is between i and j vertically
                    if(is_i_just_above_j and (pelem_i.overlaps_x(pelem_w) or pelem_j.overlaps_x(pelem_w))):
                        i_above_w:bool = pelem_i.is_strictly_above(pelem_w);
                        w_above_j:bool = pelem_w.is_strictly_above(pelem_j);
                        
                        is_i_just_above_j:bool = (not (i_above_w and w_above_j));

                if is_i_just_above_j:

                    while i in l2r_map:
                        i = l2r_map[i];

                    dn_map[i].append(j)
                    up_map[j].append(i)
                        
        return up_map, dn_map

    def _do_horizontal_dilation(self, page_elems, dilated_page_elems, up_map, dn_map):
        dilated_page_elems = page_elems # // deep-copy
        
        for i,pelem_i in enumerate(dilated_page_elems):

            x0 = pelem_i.x0;
            y0 = pelem_i.y0;

            x1 = pelem_i.x1;
            y1 = pelem_i.y1;
            
            if i in up_map and len(up_map[i])>0:
                pelem_up = page_elems[up_map[i][0]]
                
                x0 = min(x0, pelem_up.x0)
                x1 = max(x1, pelem_up.x1)

            if i in dn_map and len(dn_map[i])>0:
                pelem_dn = page_elems[dn_map[i][0]]
                
                x0 = min(x0, pelem_dn.x0)
                x1 = max(x1, pelem_dn.x1)
            
            pelem_i.x0 = x0
            pelem_i.x1 = x1
            
            overlaps_with_rest:bool = False
            for j,pelem_j in enumerate(page_elems):
            
                if i==j:
                    continue
                    
                if not overlaps_with_rest:
                    overlaps_with_rest = pelem_j.overlaps(pelem_i)
            
	    # update
            if(not overlaps_with_rest):
                dilated_page_elems[i].x0 = x0
                dilated_page_elems[i].y0 = y0
                dilated_page_elems[i].x1 = x1
                dilated_page_elems[i].y1 = y1

    def _find_heads(self, page_elems, h2i_map, i2h_map, up_map, dn_map):
        heads:list[int] = []

        head_page_elems = []
        for key,vals in up_map.items():
            if(len(vals)==0):
                head_page_elems.append(page_elems[key])

        sorted(head_page_elems) # this will invokde __lt__ from PageElements

        for item in head_page_elems:
            heads.append(h2i_map[item.cid])

        return heads
        
    def _sort_ud_maps(self, provs, h2i_map, i2h_map, up_map, dn_map):
        for ind_i,vals in dn_map.items():
            
            child_provs={}
            for ind_j in vals:
                child_provs.push_back(provs[ind_j])

            sorted(child_provs)

            dn_map[ind_i] = []
            for child in child_provs:
                dn_map[ind_i].append(h2i_map[child.cid])

    def _find_order(self, provs, heads, up_map, dn_map):
        order: list[int] = []

        visited: list[bool] = [False for _ in provs]
        
        for j in heads:

            if not visited[j]:
	        
                order.append(j)
                visited[j] = True	    
                self._depth_first_search_downwards(j, order, visited, dn_map, up_map);
                
        if len(order)!=len(provs):
            _log.error("something went wrong")

        return order

    def _depth_first_search_upwards(self, j: int,
				    order: list[int],
				    visited: list[bool],
				    dn_map: dict[int, list[int]],
				    up_map: dict[int, list[int]]):
        """depth_first_search_upwards"""
        
        k = j
        
        inds = up_map.at(j)
        for ind in inds:
            if not visited[ind]:
                return self._depth_first_search_upwards(ind, order, visited, dn_map, up_map)
    
        return k
  
    def _depth_first_search_downwards(self, j: int,
				      order: list[int],
				      visited: list[bool],
				      dn_map: dict[int, list[int]],
				      up_map: dict[int, list[int]]):
        """depth_first_search_downwards"""

        inds: list[int] = dn_map[j]

        for i in inds:
            k:int = self._depth_first_search_upwards(i, order, visited, dn_map, up_map)
	
            if not visited[k]:
                order.append(k)
                visited[k] = True

                self._depth_first_search_downwards(k, order, visited, dn_map, up_map)
