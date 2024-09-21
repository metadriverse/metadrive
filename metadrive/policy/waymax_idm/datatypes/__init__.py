# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data structures and helper operations for Waymax."""

from metadrive.policy.waymax_idm.datatypes.action import Action
from metadrive.policy.waymax_idm.datatypes.action import TrajectoryUpdate
from metadrive.policy.waymax_idm.datatypes.array import MaskedArray
from metadrive.policy.waymax_idm.datatypes.array import PyTree
from metadrive.policy.waymax_idm.datatypes.constant import TIMESTEP_MICROS_INTERVAL
from metadrive.policy.waymax_idm.datatypes.constant import TIME_INTERVAL
from metadrive.policy.waymax_idm.datatypes.object_state import ObjectMetadata
from metadrive.policy.waymax_idm.datatypes.object_state import ObjectTypeIds
from metadrive.policy.waymax_idm.datatypes.object_state import Trajectory
from metadrive.policy.waymax_idm.datatypes.object_state import fill_invalid_trajectory
from metadrive.policy.waymax_idm.datatypes.operations import compare_all_leaf_nodes
from metadrive.policy.waymax_idm.datatypes.operations import dynamic_index
from metadrive.policy.waymax_idm.datatypes.operations import dynamic_slice
from metadrive.policy.waymax_idm.datatypes.operations import dynamic_update_slice_in_dim
from metadrive.policy.waymax_idm.datatypes.operations import make_invalid_data
from metadrive.policy.waymax_idm.datatypes.operations import masked_mean
from metadrive.policy.waymax_idm.datatypes.operations import select_by_onehot
from metadrive.policy.waymax_idm.datatypes.operations import update_by_mask
from metadrive.policy.waymax_idm.datatypes.operations import update_by_slice_in_dim
from metadrive.policy.waymax_idm.datatypes.roadgraph import MapElementIds
from metadrive.policy.waymax_idm.datatypes.roadgraph import RoadgraphPoints
from metadrive.policy.waymax_idm.datatypes.roadgraph import filter_topk_roadgraph_points
from metadrive.policy.waymax_idm.datatypes.roadgraph import is_road_edge
from metadrive.policy.waymax_idm.datatypes.route import Paths
from metadrive.policy.waymax_idm.datatypes.simulator_state import SimulatorState
from metadrive.policy.waymax_idm.datatypes.simulator_state import get_control_mask
from metadrive.policy.waymax_idm.datatypes.simulator_state import update_state_by_log
from metadrive.policy.waymax_idm.datatypes.traffic_lights import TrafficLightStates
from metadrive.policy.waymax_idm.datatypes.traffic_lights import TrafficLights
