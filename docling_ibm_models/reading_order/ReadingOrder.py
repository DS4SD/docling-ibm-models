#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
from collections.abc import Iterable
from typing import Union

@dataclass
class PageElement:

    eps: float = 1.e-3
    
    cid: int # conversion id
    pid: int # page-id
    
    x0: float # lower-left x
    y0: float # lower-left y

    x1: float # upper-right x
    y1: float # upper-right y

    label: str # layout label

    def follows_maintext_order(self, rhs: PageElement) -> bool:
        return (self.cid+1==rhs.cid)

    def overlaps(self, rhs: PageElement) -> bool:
        return (self.overlaps_x(rhs) and self.overlaps_y(rhs))
    
    def overlaps_x(self, rhs: PageElement) -> bool:
        return ((self.x0<=rhs.x0 and rhs.x0<self.x1) or
	        (self.x0<=rhs.x1 and rhs.x1<self.x1) or
	        (rhs.x0<=self.x0 and self.x0<rhs.x1) or
	        (rhs.x0<=self.x1 and self.x1<rhs.x1) );
    
    def overlaps_y(self, rhs: PageElement) -> bool:
        return ((self.y0<=rhs.y0 and rhs.y0<self.y1) or
	        (self.y0<=rhs.y1 and rhs.y1<self.y1) or
	        (rhs.y0<=self.y0 and self.y0<rhs.y1) or
	        (rhs.y0<=self.y1 and self.y1<rhs.y1) );

    def overlaps_y_with_iou(self, rhs: PageElement, iou:float) -> bool:
        return False

    def is_left_of(self, rhs: PageElement) -> bool:
        return (self.x0<rhs.x0)
    
    def is_strictly_left_of(self, rhs: PageElement) -> bool:
        return (self.x1+self.eps<rhs.x0)

    """
    def is_right_of(self, rhs: PageElement) -> bool:
        return True

    def is_strictly_right_of(self, rhs: PageElement) -> bool:
        return True
    """

    """
    def is_below(self, rhs: PageElement) -> bool:
        return True
    
    def is_strictly_below(self, rhs: PageElement) -> bool:
        return True
    """
    
    def is_above(self, rhs: PageElement) -> bool:
        return (self.y0>rhs.y0)
    
    def is_strictly_above(self, rhs: PageElement) -> bool:
        (self.y0+self.eps>rhs.y1)

    def is_horizontally_connected(self, elem_i: PageElement, elem_j: PageElement) -> bool:
        min_ij:float = min(elem_i.y0, elem_j.y0)
        max_ij:float = max(elem_i.y1, elem_j.y1)

        if(self.y0<max_ij and self.y1>min_ij): # overlap_y
	    return False
    
        if(self.x0<elem_i.x1 and self.x1>elem_j.x0):
	    return True
    
    return False        
        
class ReadingOrder:
    r"""
    Rule based reading order for DoclingDocument
    """

    def __init__(self):
        self.page_elements: Dict[int, List[PageElement]] = {}

    def predict(self, conv_res: ConversionResult) -> DoclingDocument:
        r"""
        Reorder the output of the 
        """
        doc_elems = self._to_page_elements(conv_res)

        for pid, page_elems in doc_elems.items():

             h2i_map, i2h_map = self.init_h2i_map(page_elems)

             l2r_map, r2l_map = self.init_l2r_map(page_elems)

             up_map, dn_map = self.init_ud_maps(page_elems)

             heads = self.find_heads(page_elems, h2i_map, i2h_map, up_map, dn_map)
             
        doc = DoclingDocument()
        return doc

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

    def _init_h2i_map(self, page_elems: List[PageElement]):            
        h2i_map = {}
        i2h_map = {} 

        for i,pelem in enumerate(page_elems):
            h2i_map[pelem.cid] = i
            i2h_map[i] = pelem.cid
        
        return h2i_map, i2h_map
        
    def _init_l2r_map(self, page_elems: List[PageElement]):
        l2r_map = {}
        r2l_map = {} 

        for i,pelem_i in enumerate(page_elems):
            for j,pelem_j in enumerate(page_elems):        

                if(pelem_i.follows_maintext_order(pelem_j) and
                   pelem_i.is_strictly_left_of(pelem_j) and
                   pelem_i.overlaps_y(pelem_j, 0.8)):
                    l2r_map[i] = j;
                    r2l_map[j] = i;
                
        return l2r_map, r2l_map
        
    def _init_ud_maps(self, page_elems: List[PageElement]):
        up_map = {}
        dn_map = {}

        for i,pelem_i in enumerate(page_elems):
            up_map[i] = []
            dn_map[i] = []

        for j,pelem_j in enumerate(page_elems):

            if(j in r2l_map):
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

                if(is_i_just_above_j):

                    while(i in l2r_map):
                        i = l2r_map[i];

                    dn_map[i].append(j)
                    up_map[j].append(i)
                        
        return up_map, dn_map

    def find_heads(self, page_elems, h2i_map, i2h_map, up_map, dn_map):
        heads:list[int] = []

        head_provs = []
        for key,vals in up_map.items():
            if(len(vals)==0):
                head_provs.append(page_elems[key])

        sorted(head_provs, key=lambda);

        for item in head_provs.items():
            heads.append(h2i_map[item.cid))

        return heads
        
        
