#
# Copyright 2021-2023 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import datetime


class Persistence:
    def __init__(self, cvnn_path, nn_path):
        self._cvnn_path = cvnn_path
        self._cvnn_path = nn_path

    def save_cvnn_data(self, general_info, data):
        dict_data = {
            "general_info": general_info,
            "data": data,
        }

        file_name_postfix = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        with open(
            "scripts/gate_synthesis/cvnn_approximations/train_data{}.json".format(
                file_name_postfix
            ),
            "w",
        ) as json_file:
            json.dump(dict_data, json_file, indent=4)

        print(
            "Datapack saved by the name: {}".format(
                "train_data" + file_name_postfix + ".json"
            )
        )
