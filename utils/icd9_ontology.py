import pandas as pd
import re
from utils.icd9 import ICD9


class SecondLevelCodes(object):
    def __init__(self,icd_jon):
        tree = ICD9(icd_jon)
        # list of top level codes (e.g., '001-139', ...)
        top_level_nodes = tree.children

        # second level
        self.second_level_codes = []
        for node in top_level_nodes:
            children = node.children
            codes = [node.code for node in children]
            self.second_level_codes.extend(codes)

    def second_level_codes_icd9(self, dxStr):
        code_3digit = convert_to_3digit_icd9(dxStr)
        for code in self.second_level_codes:
            if len(code) > 4:
                codes = code.split('-')
                if codes[0] <= code_3digit <= codes[1]:
                    return code
            elif code == code_3digit:
                return code


def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3]
        else:
            return dxStr


def convert_to_high_level_icd9(dxStr):

    k = dxStr[:3]
    if '001' <= k <= '139':
        return 0
    elif '140' <= k <= '239':
        return 1
    elif '240' <= k <= '279':
        return 2
    elif '280' <= k <= '289':
        return 3
    elif '290' <= k <= '319':
        return 4
    elif '320' <= k <= '389':
        return 5
    elif '390' <= k <= '459':
        return 6
    elif '460' <= k <= '519':
        return 7
    elif '520' <= k <= '579':
        return 8
    elif '580' <= k <= '629':
        return 9
    elif '630' <= k <= '679':
        return 10
    elif '680' <= k <= '709':
        return 11
    elif '710' <= k <= '739':
        return 12
    elif '740' <= k <= '759':
        return 13
    elif '760' <= k <= '779':
        return 14
    elif '780' <= k <= '799':
        return 15
    elif '800' <= k <= '999':
        return 16
    elif 'E00' <= k <= 'E99':
        return 17
    elif 'V01' <= k <= 'V90':
        return 18


class ICD_Ontology():
    
    def __init__(self,icd_file, dx_flag):
        self.icd_file = icd_file
        self.dx_flag = dx_flag
        self.df = pd.read_csv(self.icd_file, index_col=0, dtype=object)
        self.rootLevel()



    def rootLevel(self):

        dxs = self.df.ICD9_CODE.tolist()
        dxMaps = dict()
        
        if self.dx_flag:
            
            for dx in dxs:
                dxMaps.setdefault(dx[0:3], 0)

            for k in dxMaps.keys():
                if '001' <= k <= '139':
                    dxMaps[k] = 1
                if '140' <= k <= '239':
                    dxMaps[k] = 2
                if '240' <= k <= '279':
                    dxMaps[k] = 3
                if '280' <= k <= '289':
                    dxMaps[k] = 4
                if '290' <= k <= '319':
                    dxMaps[k] = 5
                if '320' <= k <= '389':
                    dxMaps[k] = 6
                if '390' <= k <= '459':
                    dxMaps[k] = 7
                if '460' <= k <= '519':
                    dxMaps[k] = 8
                if '520' <= k <= '579':
                    dxMaps[k] = 9
                if '580' <= k <= '629':
                    dxMaps[k] = 10
                if '630' <= k <= '679':
                    dxMaps[k] = 11
                if '680' <= k <= '709':
                    dxMaps[k] = 12
                if '710' <= k <= '739':
                    dxMaps[k] = 13
                if '740' <= k <= '759':
                    dxMaps[k] = 14
                if '760' <= k <= '779':
                    dxMaps[k] = 15
                if '780' <= k <= '799':
                    dxMaps[k] = 16
                if '800' <= k <= '999':
                    dxMaps[k] = 17
                if 'E00' <= k <= 'E99':
                    dxMaps[k] = 18
                if 'V01' <= k <= 'V90':
                    dxMaps[k] = 19
            self.rootMaps = dxMaps
            
        else:
            
            for dx in dxs:
                dxMaps.setdefault(dx[0:2], 0)

            for k in dxMaps.keys():
                if k == '00':
                    dxMaps[k] = 1
                if '01' <= k <= '05':
                    dxMaps[k] = 2
                if '06' <= k <= '07':
                    dxMaps[k] = 3
                if '08' <= k <= '16':
                    dxMaps[k] = 4
                if k == '17':
                    dxMaps[k] = 5
                if '18' <= k <= '20':
                    dxMaps[k] = 6
                if '21' <= k <= '29':
                    dxMaps[k] = 7
                if '30' <= k <= '34':
                    dxMaps[k] = 8
                if '35' <= k <= '39':
                    dxMaps[k] = 9
                if '40' <= k <= '41':
                    dxMaps[k] = 10
                if '42' <= k <= '54':
                    dxMaps[k] = 11
                if '55' <= k <= '59':
                    dxMaps[k] = 12
                if '60' <= k <= '64':
                    dxMaps[k] = 13
                if '65' <= k <= '71':
                    dxMaps[k] = 14
                if '72' <= k <= '75':
                    dxMaps[k] = 15
                if '76' <= k <= '84':
                    dxMaps[k] = 16
                if '85' <= k <= '86':
                    dxMaps[k] = 17
                if '87' <= k <= '99':
                    dxMaps[k] = 18
            dxMaps['E'] = 19
            dxMaps['V'] = 20
            self.rootMaps = dxMaps

    def getRootLevel(self,code):
        
        if self.dx_flag:
            root = code[0:3]
        else:
            if code.startswith('E'):
                root = 'E'
            elif code.startswith('V'):
                root = 'V'
            else:
                root = code[0:2]
        return self.rootMaps[root]


class CCS_Ontology(object):
    
    def __init__(self, ccs_file):
        self.ccs_file = ccs_file
        self.rootLevel()
        
    def rootLevel(self):
        # ccs_file = '../data/CCS/SingleDX-edit.txt'
        with open(self.ccs_file) as f:
            content = f.readlines()

        pattern_code = '^\w+'  # match code line in file
        pattern_newline = '^\n'  # match new line '\n'

        prog_code = re.compile(pattern_code)
        prog_newline = re.compile(pattern_newline)

        catIndex = 0
        catMap = dict()  # store index:code list
        codeList = list()
        for line in content:
            
            # if the current line is code line, parse codes to a list and add to existing code list.
            result_code = prog_code.match(line)
            if result_code:
                codes = line.split()
                codeList.extend(codes)

            # if current line is a new line, add new index and corresponding code list to the catMap dict.
            result_newline = prog_newline.match(line)
            if result_newline:
                catMap[catIndex] = codeList
                codeList = list()  # initualize the code list to empty
                catIndex += 1  # next index
                
        code2CatMap = dict()
        for key, value in catMap.items():
            for code in value:
                code2CatMap.setdefault(code, key)

        self.rootMaps = code2CatMap
    
    def getRootLevel(self, code):
        return self.rootMaps[code]
