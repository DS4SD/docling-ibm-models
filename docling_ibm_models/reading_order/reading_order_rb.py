#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import copy
import os
from collections.abc import Iterable
from typing import Dict, List

from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel


class PageElement(BaseModel):

    eps: float = 1.0e-3

    cid: int

    page_no: int
    
    page_width: float
    page_height: float 

    bbox: BoundingBox

    label: DocItemLabel

    def __str__(self):
        return f"{self.cid:3.2f}\t{str(self.label):<10}\t{self.bbox.l:3.2f}, {self.bbox.b:3.2f}, {self.bbox.r:3.2f}, {self.bbox.t:3.2f}"

    def __lt__(self, other):
        if self.page_no == other.page_no:

            if self.overlaps_x(other):
                return self.bbox.b > other.bbox.b
            else:
                return self.bbox.l < other.bbox.l
        else:
            return self.page_no < other.page_no

    def follows_maintext_order(self, rhs) -> bool:
        return self.cid + 1 == rhs.cid

    def overlaps(self, rhs: "PageElement") -> bool:
        return self.overlaps_x(rhs) and self.overlaps_y(rhs)

    def overlaps_x(self, rhs: "PageElement") -> bool:
        return (
            (self.bbox.l <= rhs.bbox.l and rhs.bbox.l < self.bbox.r)
            or (self.bbox.l <= rhs.bbox.r and rhs.bbox.r < self.bbox.r)
            or (rhs.bbox.l <= self.bbox.l and self.bbox.l < rhs.bbox.r)
            or (rhs.bbox.l <= self.bbox.r and self.bbox.r < rhs.bbox.r)
        )

    def overlaps_y(self, rhs: "PageElement") -> bool:
        return (
            (self.bbox.b <= rhs.bbox.b and rhs.bbox.b < self.bbox.t)
            or (self.bbox.b <= rhs.bbox.t and rhs.bbox.t < self.bbox.t)
            or (rhs.bbox.b <= self.bbox.b and self.bbox.b < rhs.bbox.t)
            or (rhs.bbox.b <= self.bbox.t and self.bbox.t < rhs.bbox.t)
        )

    def overlaps_y_with_iou(self, rhs: "PageElement", iou: float) -> bool:
        if self.overlaps_y(rhs):

            u0 = min(self.bbox.b, rhs.bbox.b)
            u1 = max(self.bbox.t, rhs.bbox.t)

            i0 = max(self.bbox.b, rhs.bbox.b)
            i1 = min(self.bbox.t, rhs.bbox.t)

            iou_ = float(i1 - i0) / float(u1 - u0)
            return (iou_) > iou
        
        return False

    def is_left_of(self, rhs: "PageElement") -> bool:
        return self.bbox.l < rhs.bbox.l

    def is_strictly_left_of(self, rhs: "PageElement") -> bool:
        return (self.bbox.r + self.eps) < rhs.bbox.l

    """"
    def is_above(self, rhs: "PageElement") -> bool:
        return self.bbox.b > rhs.bbox.b
    """

    def is_above(self, rhs: "PageElement") -> bool:
        return self.bbox.t > rhs.bbox.t
    
    def is_strictly_above(self, rhs: "PageElement") -> bool:
        return (self.bbox.b + self.eps) > rhs.bbox.t

    def is_horizontally_connected(
        self, elem_i: "PageElement", elem_j: "PageElement"
    ) -> bool:
        min_ij: float = min(elem_i.bbox.b, elem_j.bbox.b)
        max_ij: float = max(elem_i.bbox.t, elem_j.bbox.t)

        if self.bbox.b < max_ij and self.bbox.t > min_ij:  # overlap_y
            return False

        if self.bbox.l < elem_i.bbox.r and self.bbox.r > elem_j.bbox.l:
            return True

        return False


class ReadingOrderPredictor:
    r"""
    Rule based reading order for DoclingDocument
    """

    def __init__(self):
        self.dilated_page_element = True

        self.initialise()

    def initialise(self):
        self.h2i_map: Dict[int, int] = {}
        self.i2h_map: Dict[int, int] = {}

        self.l2r_map: Dict[int, List[int]] = {}
        self.r2l_map: Dict[int, List[int]] = {}

        self.up_map: Dict[int, List[int]] = {}
        self.dn_map: Dict[int, List[int]] = {}

        self.heads: List[int] = []
        
    def predict_reading_order(
        self, page_elements: List[PageElement]
    ) -> List[PageElement]:

        page_nos: Set[int] = set() 

        for i, elem in enumerate(page_elements):
            page_nos.add(elem.page_no)
        
        page_to_elems: Dict[int, List[PageElement]] = {}
        page_to_headers: Dict[int, List[PageElement]] = {}
        page_to_footers: Dict[int, List[PageElement]] = {}
        
        for page_no in page_nos:
            page_to_elems[page_no] = []
            page_to_footers[page_no] = []
            page_to_headers[page_no] = []

        for i, elem in enumerate(page_elements):            
            if elem.label==DocItemLabel.PAGE_HEADER:
                page_to_headers[page_no].append(elem)
            elif elem.label==DocItemLabel.PAGE_FOOTER:
                page_to_footers[page_no].append(elem)
            else:            
                page_to_elems[page_no].append(elem)

        for page_no, elems in page_to_elems.items():            
            page_to_elems[page_no] = predict_page(elems)
            
        for page_no, elems in page_to_headers.items():
            page_to_headers[page_no] = predict_page(elems)
            
        for page_no, elems in page_to_footers.items():
            page_to_footers[page_no] = predict_page(elems)

        sorted_elements = []
        for page_no in page_nos:
            for k,v in page_to_headers[page_no].items():
                sorted_elements.append(v)
            for k,v in page_to_elems[page_no].items():
                sorted_elements.append(v)                
            for k,v in page_to_footers[page_no].items():
                sorted_elements.append(v)

        return sorted_elements

    def predict_to_captions(self, sorted_elements: List[PageElement]) -> Dict[int, List[int]]:

        to_captions: Dict[int, List[int]] = {}

        page_nos: Set[int] = set() 
        for i, elem in enumerate(sorted_elements):
            page_nos.add(elem.page_no)

        for page_no in page_nos:
            page_to_elems[page_no] = []

        for i, elem in enumerate(sorted_elements):            
            page_to_elems[page_no].append(elem)

        for page_no, elems in page_to_elems.items():            
            self._find_to_captions(page_elements=page_to_elems[page_no])
            for key,val in self.to_captions.items():
                to_captions[key.cid] = [_.cid for _ in val]
            
        return to_captions

    def predict_to_footnotes(self, sorted_elements: List[PageElement]) -> Dict[int, List[int]]:

        to_footnotes: Dict[int, List[int]] = {}

        page_nos: Set[int] = set() 
        for i, elem in enumerate(sorted_elements):
            page_nos.add(elem.page_no)

        for page_no in page_nos:
            page_to_elems[page_no] = []

        for i, elem in enumerate(sorted_elements):            
            page_to_elems[page_no].append(elem)

        for page_no, elems in page_to_elems.items():            
            self._find_to_footnotes(page_elements=page_to_elems[page_no])
            for key,val in self.to_footnotes.items():
                to_footnotes[key.cid] = [_.cid for _ in val]
            
        return to_footnotes    

    def predict_to_merges(self, sorted_elements: List[PageElement]) -> Dict[int, List[int]]:

        merges: Dict[int, List[int]] = {}

        page_nos: Set[int] = set() 
        for i, elem in enumerate(sorted_elements):
            page_nos.add(elem.page_no)

        for page_no in page_nos:
            page_to_elems[page_no] = []

        for i, elem in enumerate(sorted_elements):
            if elem.label not in [DocItemLabel.PAGE_HEADER,
                                  DocItemLabel.PAGE_FOOTER,
                                  DocItemLabel.FOOTNOTE,
                                  DocItemLabel.TABLE,
                                  DocItemLabel.IMAGE]:
                page_to_elems[page_no].append(elem)        

        return merges
                
    def predict_page(
        self, page_elements: List[PageElement]
    ) -> List[PageElement]:
        r"""
        Reorder the output of the
        """

        self.initialise()

        for i, elem in enumerate(page_elements):
            page_elements[i].bbox = elem.bbox.to_bottom_left_origin(
                page_height=elem.page_height
            )

        self._init_h2i_map(page_elements)

        self._init_l2r_map(page_elements)

        self._init_ud_maps(page_elements)

        if self.dilated_page_element:
            dilated_page_elements: List[PageElement] = copy.deepcopy(
                page_elements
            )  # deep-copy
            dilated_page_elements = self._do_horizontal_dilation(
                page_elements, dilated_page_elements
            )

            # redo with dilated provs
            self._init_ud_maps(dilated_page_elements)

        self._find_heads(page_elements)

        self._sort_ud_maps(page_elements)
        order: List[int] = self._find_order(page_elements)

        sorted_page_elements: List[PageElement] = []
        for ind in order:
            sorted_page_elements.append(page_elements[ind])

        #self._find_to_captions(page_elements=sorted_page_elements)
        
        return sorted_page_elements#, self.to_captions, self.to_footnotes

    def _init_h2i_map(self, page_elems: List[PageElement]):
        self.h2i_map = {}
        self.i2h_map = {}

        for i, pelem in enumerate(page_elems):
            self.h2i_map[pelem.cid] = i
            self.i2h_map[i] = pelem.cid

    def _init_l2r_map(self, page_elems: List[PageElement]):
        self.l2r_map = {}
        self.r2l_map = {}

        for i, pelem_i in enumerate(page_elems):
            for j, pelem_j in enumerate(page_elems):

                if (
                    pelem_i.follows_maintext_order(pelem_j)
                    and pelem_i.is_strictly_left_of(pelem_j)
                    and pelem_i.overlaps_y_with_iou(pelem_j, 0.8)
                ):
                    self.l2r_map[i] = j
                    self.r2l_map[j] = i

    def _init_ud_maps(self, page_elems: List[PageElement]):
        self.up_map = {}
        self.dn_map = {}

        for i, pelem_i in enumerate(page_elems):
            self.up_map[i] = []
            self.dn_map[i] = []

        for j, pelem_j in enumerate(page_elems):

            if j in self.r2l_map:
                i = self.r2l_map[j]

                self.dn_map[i] = [j]
                self.up_map[j] = [i]

                continue

            for i, pelem_i in enumerate(page_elems):

                if i == j:
                    continue

                is_horizontally_connected: bool = False
                is_i_just_above_j: bool = pelem_i.overlaps_x(
                    pelem_j
                ) and pelem_i.is_strictly_above(pelem_j)

                for w, pelem_w in enumerate(page_elems):

                    if not is_horizontally_connected:
                        is_horizontally_connected = pelem_w.is_horizontally_connected(
                            pelem_i, pelem_j
                        )

                    # ensure there is no other element that is between i and j vertically
                    if is_i_just_above_j and (
                        pelem_i.overlaps_x(pelem_w) or pelem_j.overlaps_x(pelem_w)
                    ):
                        i_above_w: bool = pelem_i.is_strictly_above(pelem_w)
                        w_above_j: bool = pelem_w.is_strictly_above(pelem_j)

                        is_i_just_above_j: bool = not (i_above_w and w_above_j)

                if is_i_just_above_j:

                    while i in self.l2r_map:
                        i = self.l2r_map[i]

                    self.dn_map[i].append(j)
                    self.up_map[j].append(i)

    def _do_horizontal_dilation(self, page_elems, dilated_page_elems):

        for i, pelem_i in enumerate(dilated_page_elems):

            x0 = pelem_i.bbox.l
            y0 = pelem_i.bbox.b

            x1 = pelem_i.bbox.r
            y1 = pelem_i.bbox.t

            if i in self.up_map and len(self.up_map[i]) > 0:
                pelem_up = page_elems[self.up_map[i][0]]

                x0 = min(x0, pelem_up.bbox.l)
                x1 = max(x1, pelem_up.bbox.r)

            if i in self.dn_map and len(self.dn_map[i]) > 0:
                pelem_dn = page_elems[self.dn_map[i][0]]

                x0 = min(x0, pelem_dn.bbox.l)
                x1 = max(x1, pelem_dn.bbox.r)

            pelem_i.bbox.l = x0
            pelem_i.bbox.r = x1

            overlaps_with_rest: bool = False
            for j, pelem_j in enumerate(page_elems):

                if i == j:
                    continue

                if not overlaps_with_rest:
                    overlaps_with_rest = pelem_j.overlaps(pelem_i)

            # update
            if not overlaps_with_rest:
                dilated_page_elems[i].bbox.l = x0
                dilated_page_elems[i].bbox.b = y0
                dilated_page_elems[i].bbox.r = x1
                dilated_page_elems[i].bbox.t = y1

        return dilated_page_elems

    def _find_heads(self, page_elems: List[PageElement]):
        # heads:List[int] = []

        head_page_elems = []
        for key, vals in self.up_map.items():
            if len(vals) == 0:
                head_page_elems.append(page_elems[key])

        sorted(head_page_elems)  # this will invokde __lt__ from PageElements

        for item in head_page_elems:
            self.heads.append(self.h2i_map[item.cid])

    def _sort_ud_maps(self, provs: List[PageElement]):
        for ind_i, vals in self.dn_map.items():

            child_provs: List[PageElement] = []
            for ind_j in vals:
                child_provs.append(provs[ind_j])

            sorted(child_provs)

            self.dn_map[ind_i] = []
            for child in child_provs:
                self.dn_map[ind_i].append(self.h2i_map[child.cid])

    def _find_order(self, provs: List[PageElement]):
        order: List[int] = []

        visited: List[bool] = [False for _ in provs]

        for j in self.heads:

            if not visited[j]:

                order.append(j)
                visited[j] = True
                self._depth_first_search_downwards(j, order, visited)

        if len(order) != len(provs):
            _log.error("something went wrong")

        return order

    def _depth_first_search_upwards(
        self, j: int, order: List[int], visited: List[bool]
    ):
        """depth_first_search_upwards"""

        k = j

        inds = self.up_map[j]
        for ind in inds:
            if not visited[ind]:
                return self._depth_first_search_upwards(ind, order, visited)

        return k

    def _depth_first_search_downwards(
        self, j: int, order: List[int], visited: List[bool]
    ):
        """depth_first_search_downwards"""

        inds: List[int] = self.dn_map[j]

        for i in inds:
            k: int = self._depth_first_search_upwards(i, order, visited)

            if not visited[k]:
                order.append(k)
                visited[k] = True

                self._depth_first_search_downwards(k, order, visited)

    def _find_to_captions(self, page_elements: List[PageElement]) -> Dict[int, List[int]]:
        
        captions: Set[int] = set()
        
        to_captions: Dict[int, List[int]] = {}

        # Try find captions that precede the table and footnotes that come after the table
        for ind, page_element in enumerate(page_elements):

            if page_element.label == DocItemLabel.TABLE:

                ind_j = ind - 1
                if ind_j >= 0 and page_elements[ind_j].label == DocItemLabel.CAPTION:
                    captions.add(ind_j)
                    if ind in self.to_captions:
                        self.to_captions[ind].append(ind_j)
                    else:
                        self.to_captions[ind] = [ind_j]

        # Try find captions that precede the table and footnotes that come after the table
        for ind, page_element in enumerate(page_elements):

            if page_element.label == DocItemLabel.PICTURE:

                ind_j = ind + 1
                if (
                    ind_j < len(page_elements)
                    and page_elements[ind_j].label == DocItemLabel.CAPTION
                ):
                    captions.add(ind_j)
                    if ind in self.to_captions:
                        self.to_captions[ind].append(ind_j)
                    else:
                        self.to_captions[ind] = [ind_j]


    def _find_to_footnotes(self, page_elements: List[PageElement]) -> Dict[int, List[int]]:

        footnotes: Set[int] = set()

        to_footnotes: Dict[int, List[int]] = {}

        # Try find captions that precede the table and footnotes that come after the table
        for ind, page_element in enumerate(page_elements):

            if page_element.label == DocItemLabel.TABLE:

                ind_j = ind + 1
                while (
                    ind_j < len(page_elements)
                    and page_elements[ind_j].label == DocItemLabel.FOOTNOTE
                ):
                    footnotes.add(ind_j)
                    if ind in self.to_footnotes:
                        self.to_footnotes[ind].append(ind_j)
                    else:
                        self.to_footnotes[ind] = [ind_j]

                    ind_j += 1

        # Try find captions that precede the table and footnotes that come after the table
        for ind, page_element in enumerate(page_elements):

            if page_element.label == DocItemLabel.PICTURE:

                ind_j = ind + 1
                while (
                    ind_j < len(page_elements)
                    and page_elements[ind_j].label == DocItemLabel.FOOTNOTE
                ):
                    footnotes.add(ind_j)
                    if ind in self.to_footnotes:
                        self.to_footnotes[ind].append(ind_j)
                    else:
                        self.to_footnotes[ind] = [ind_j]

                    ind_j += 1

                    
