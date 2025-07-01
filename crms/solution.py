import os
import json
from datetime import datetime
import c_two as cc
from pathlib import Path
from icrms.isolution import ISolution,NeData,NsData,RainfallData,TideData,HumanAction,Gate,GridResult
import logging
logger = logging.getLogger(__name__)

@cc.iicrm
class Solution(ISolution):
    def __init__(self,path: Path,imp_path: Path,ne_path: Path,ns_path: Path,rainfall_path: Path, gate_path: Path,tide_path: Path):
        self.path = path
        self.imp_path = imp_path
        self.ne_path = ne_path
        self.ns_path = ns_path
        self.rainfall_path = rainfall_path
        self.gate_path = gate_path
        self.tide_path = tide_path

        human_action_path = path / 'human_action'
        self.human_action_path = human_action_path
        self.human_action_path.mkdir(parents=True, exist_ok=True)

        result_path = path / 'result'
        self.result_path = result_path
        self.result_path.mkdir(parents=True, exist_ok=True)
            
    def get_imp(self) -> str:
        with open(self.imp_path, 'r', encoding='utf-8') as f:
            data = f.read()
        return data
    
    def get_ne(self) -> NeData:
        grid_id_list = []
        nsl1_list = []
        nsl2_list = []
        nsl3_list = []
        nsl4_list = []
        isl1_list = []
        isl2_list = []
        isl3_list = []
        isl4_list = []
        xe_list = []
        ye_list = []
        ze_list = []
        under_suf_list = []
        with open(self.or_ne, 'r', encoding='utf-8') as f:
            for row_data in f:
                row_data = row_data.split(',')
                # 创建NeData对象
                grid_id_list.append(int(row_data[0]))
                nsl1 = int(row_data[1])
                nsl2 = int(row_data[2])
                nsl3 = int(row_data[3])
                nsl4 = int(row_data[4])
                nsl1_list.append(nsl1)
                nsl2_list.append(nsl2)
                nsl3_list.append(nsl3)
                nsl4_list.append(nsl4)
                isl1 = [0 for _ in range(nsl1)]
                isl2 = [0 for _ in range(nsl2)]
                isl3 = [0 for _ in range(nsl3)]
                isl4 = [0 for _ in range(nsl4)]
                for i in range(nsl1): 
                    isl1[i] = int(row_data[5+i]) 
                for i in range(nsl2): 
                    isl2[i] = int(row_data[5+nsl1+i])
                for i in range(nsl3): 
                    isl3[i] = int(row_data[5+nsl1+nsl2+i])
                for i in range(nsl4): 
                    isl4[i] = int(row_data[5+nsl1+nsl2+nsl3+i])
                isl1_list.append(isl1)
                isl2_list.append(isl2)
                isl3_list.append(isl3)
                isl4_list.append(isl4)
                xe_list.append(float(row_data[-4]))
                ye_list.append(float(row_data[-3]))
                ze_list.append(float(row_data[-2]))
                under_suf_list.append(int(row_data[-1]))
                break       
        ne_data = NeData(grid_id_list,nsl1_list,nsl2_list,nsl3_list,nsl4_list,isl1_list,isl2_list,isl3_list,isl4_list,xe_list,ye_list,ze_list,under_suf_list)
        return ne_data
    
    def get_ns(self) -> NsData:
        edge_id = []
        ise = []
        dis = []
        x_side = []
        y_side = []
        z_side = []
        under_suf = []
        nbd_ie = []
        ibd_ie = []
        with open(self.or_ns,'r',encoding='utf-8') as f:
            for rowdata in f:
                rowdata = rowdata.strip().split(",")
                edge_id.append(int(float(rowdata[0].strip())))
                ise.append(int(float(rowdata[1].strip())))
                dis.append(float(rowdata[2].strip()))
                x_side.append(float(rowdata[3].strip()))
                y_side.append(float(rowdata[4].strip()))
                z_side.append(float(rowdata[5].strip()))
                under_suf.append(float(rowdata[6].strip()))
                nbd_ie.append(int(float(rowdata[7].strip())))
                ibd_ie.append(int(float(rowdata[8].strip())))
                break
        ns_data = NsData(
            edge_id,
            ise,
            dis,
            x_side,
            y_side,
            z_side,
            under_suf,
            nbd_ie,
            ibd_ie
        )
        return ns_data
    
    def get_rainfall(self) -> RainfallData:
        rainfall_date_list = []
        rainfall_station_list = []
        rainfall_value_list = []
        with open(self.rainfall_path,'r',encoding='utf-8') as f:
            # 跳过第一行
            next(f)
            for row_data in f:
                row_data = row_data.split(',')
                rainfall_date_list.append(row_data[0])
                rainfall_station_list.append(row_data[1])
                rainfall_value_list.append(float(row_data[2]))
                break
        rainfall = RainfallData(
            rainfall_date_list,
            rainfall_station_list,
            rainfall_value_list
        )
        return rainfall
    
    def get_gate(self) -> Gate:
        gate_id_list = []
        gate_num_list = []
        grid_id_list = []
        ud_stream_list = []
        gate_height_list = []
        with open(self.gate_path,'r',encoding='utf-8') as f:
            for row_data in f:
                row_data = row_data.strip().split(',')
                gate_id_list.append(int(row_data[0]))
                gate_num_list.append(int(row_data[1]))
                for i in range(2, 2 + int(row_data[1])):
                    grid_id_list.append(int(row_data[i]))
                ud_stream_list.append(int(row_data[-3]))
                ud_stream_list.append(int(row_data[-2]))
                gate_height_list.append(int(row_data[-1]))
                # break
        gate = Gate(
            gate_id_list=gate_id_list,
            gate_num_list=gate_num_list,
            grid_id_list=grid_id_list,
            ud_stream_list=ud_stream_list,
            gate_height_list=gate_height_list
        )
        return gate
    
    def get_tide(self) -> TideData:
        tide_date_list = []
        tide_time_list = []
        tide_value_list = []
        with open(self.tide_path,'r',encoding='utf-8') as f:
            # 跳过第一行
            next(f)
            for row_data in f:
                row_data = row_data.split(',')
                tide_date_list.append(row_data[0])
                tide_time_list.append(row_data[1])
                tide_value_list.append(float(row_data[2]))
                break
        tide = TideData(
            tide_date_list,
            tide_time_list,
            tide_value_list
        )
        return tide
 
    def terminate(self) -> None:
        # Do something need to be saved
        pass
    
    def get_human_actions(self, step: int) -> list[HumanAction]:
        step_path = self.human_action_path / str(step)
        action_files = step_path.glob('*.json')
        actions = []

        # 按时间排序，基于文件名中的时间戳
        action_files = sorted(action_files, key=lambda x: datetime.strptime(x.stem.split('_')[-1], "%Y-%m-%d-%H-%M-%S-%f"))
        
        for action_file in action_files:
            with open(action_file, 'r', encoding='utf-8') as f:
                action = HumanAction.model_validate_json(f.read())
                actions.append(action)
        
        return actions

    def send_result(self, step: int, result: list[GridResult], highlight_grids: list[int]) -> dict[str, bool | str]:
        try:
            step_path = self.result_path / str(step)
            step_path.mkdir(parents=True, exist_ok=True)
            result_path = step_path / 'result.json'
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            if highlight_grids:
                highlight_path = step_path / 'highlight_grids.json'
                with open(highlight_path, 'w', encoding='utf-8') as f:
                    json.dump(highlight_grids, f, ensure_ascii=False, indent=4)
            return {'success': True, 'message': 'success'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    # ------------------------------------------------------------
    # Front to Resource Server
    def add_human_action(self, step: int, action: HumanAction) -> dict[str, bool | str]:
        try:
            step_path = self.human_action_path / str(step)
            step_path.mkdir(parents=True, exist_ok=True)

            # 使用毫秒级别的时间戳生成唯一时间标识
            time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            action_path = step_path / f'action_{time}.json'

            # 存储时直接使用 datetime 对象
            with open(action_path, 'w', encoding='utf-8') as f:
                json.dump(action.model_dump(), f, ensure_ascii=False, indent=4)

            return {'success': True, 'message': 'success'}
        except Exception as e:
            return {'success': False, 'message': str(e)}