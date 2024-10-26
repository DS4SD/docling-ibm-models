#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
from collections.abc import Iterable
from typing import Union

@dataclass
class PageElement:

    cid: int # conversion id
    pid: int # page-id
    
    x0: float # lower-left x
    y0: float # lower-left y

    x1: float # upper-right x
    y1: float # upper-right y

    label: str # layout label

    def overlaps_x(self, other: PageElement) -> bool:
        return True
    
    def overlaps_y(self, other: PageElement) -> bool:
        return True

    def is_strictly_left_of(self, other: PageElement) -> bool:
        return True
    
    def is_strictly_right_of(self, other: PageElement) -> bool:
        return True

    def is_strictly_below(self, other: PageElement) -> bool:
        return True
    
    def is_strictly_above(self, other: PageElement) -> bool:
        return True

    def follows_maintext_order(self, other: PageElement) -> bool:
        return True
    
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

             h2i_map, i2h_map = init_h2i_map(page_elems)

             l2r_map, r2l_map = init_l2r_map(page_elems)

             up_map, dn_map = init_ud_maps(page_elems)
             
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

    def _init_h2i_map(self, page_elems):            
        h2i_map = {}
        i2h_map = {} 

        for i,pelem in enumerate(page_elems):
            h2i_map[pelem.cid] = i
            i2h_map[i] = pelem.cid
        
        return h2i_map, i2h_map
        
    def _init_l2r_map(self, page_elems):
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
        
    def _init_ud_maps(self, page_elems):
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
