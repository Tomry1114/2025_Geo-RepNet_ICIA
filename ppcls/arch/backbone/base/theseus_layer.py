# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, List, Dict, Union, Callable, Any

from paddle import nn
from ....utils import logger


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.res_dict = {}
        self.res_name = self.full_name()
        self.pruner = None
        self.quanter = None

        self.init_net(*args, **kwargs)

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"logits": output}
        # 'list' is needed to avoid error raised by popping self.res_dict
        for res_key in list(self.res_dict):
            # clear the res_dict because the forward process may change according to input
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def init_net(self,
                 stages_pattern=None,
                 return_patterns=None,
                 return_stages=None,
                 freeze_befor=None,
                 stop_after=None,
                 *args,
                 **kwargs):
        # init the output of net
        if return_patterns or return_stages:
            if return_patterns and return_stages:
                msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."
                logger.warning(msg)
                return_stages = None

            if return_stages is True:
                return_patterns = stages_pattern

            # return_stages is int or bool
            if type(return_stages) is int:
                return_stages = [return_stages]
            if isinstance(return_stages, list):
                if max(return_stages) > len(stages_pattern) or min(
                        return_stages) < 0:
                    msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                    logger.warning(msg)
                    return_stages = [
                        val for val in return_stages
                        if val >= 0 and val < len(stages_pattern)
                    ]
                return_patterns = [stages_pattern[i] for i in return_stages]

            if return_patterns:
                # call update_res function after the __init__ of the object has completed execution, that is, the contructing of layer or model has been completed.
                def update_res_hook(layer, input):
                    self.update_res(return_patterns)

                self.register_forward_pre_hook(update_res_hook)

        # freeze subnet
        if freeze_befor is not None:
            self.freeze_befor(freeze_befor)

        # set subnet to Identity
        if stop_after is not None:
            self.stop_after(stop_after)

    def init_res(self,
                 stages_pattern,
                 return_patterns=None,
                 return_stages=None):
        msg = "\"init_res\" will be deprecated, please use \"init_net\" instead."
        logger.warning(DeprecationWarning(msg))

        if return_patterns and return_stages:
            msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."
            logger.warning(msg)
            return_stages = None

        if return_stages is True:
            return_patterns = stages_pattern
        # return_stages is int or bool
        if type(return_stages) is int:
            return_stages = [return_stages]
        if isinstance(return_stages, list):
            if max(return_stages) > len(stages_pattern) or min(
                    return_stages) < 0:
                msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                logger.warning(msg)
                return_stages = [
                    val for val in return_stages
                    if val >= 0 and val < len(stages_pattern)
                ]
            return_patterns = [stages_pattern[i] for i in return_stages]

        if return_patterns:
            self.update_res(return_patterns)

    def replace_sub(self, *args, **kwargs) -> None:
        msg = "The function 'replace_sub()' is deprecated, please use 'upgrade_sublayer()' instead."
        logger.error(DeprecationWarning(msg))
        raise DeprecationWarning(msg)

    def upgrade_sublayer(self,
                         layer_name_pattern: Union[str, List[str]],
                         handle_func: Callable[[nn.Layer, str], nn.Layer]
                         ) -> Dict[str, nn.Layer]:
        """use 'handle_func' to modify the sub-layer(s) specified by 'layer_name_pattern'.

        Args:
            layer_name_pattern (Union[str, List[str]]): The name of layer to be modified by 'handle_func'.
            handle_func (Callable[[nn.Layer, str], nn.Layer]): The function to modify target layer specified by 'layer_name_pattern'. The formal params are the layer(nn.Layer) and pattern(str) that is (a member of) layer_name_pattern (when layer_name_pattern is List type). And the return is the layer processed.

        Returns:
            Dict[str, nn.Layer]: The key is the pattern and corresponding value is the result returned by 'handle_func()'.

        Examples:

            from paddle import nn
            import paddleclas

            def rep_func(layer: nn.Layer, pattern: str):
                new_layer = nn.Conv2D(
                    in_channels=layer._in_channels,
                    out_channels=layer._out_channels,
                    kernel_size=5,
                    padding=2
                )
                return new_layer

            net = paddleclas.MobileNetV1()
            res = net.upgrade_sublayer(layer_name_pattern=["blocks[11].depthwise_conv.conv", "blocks[12].depthwise_conv.conv"], handle_func=rep_func)
            print(res)
            # {'blocks[11].depthwise_conv.conv': the corresponding new_layer, 'blocks[12].depthwise_conv.conv': the corresponding new_layer}
        """

        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]

        hit_layer_pattern_list = []
        for pattern in layer_name_pattern:
            # parse pattern to find target layer and its parent
            layer_list = parse_pattern_str(pattern=pattern, parent_layer=self)
            if not layer_list:
                continue

            sub_layer_parent = layer_list[-2]["layer"] if len(
                layer_list) > 1 else self
            sub_layer = layer_list[-1]["layer"]
            sub_layer_name = layer_list[-1]["name"]
            sub_layer_index_list = layer_list[-1]["index_list"]

            new_sub_layer = handle_func(sub_layer, pattern)

            if sub_layer_index_list:
                if len(sub_layer_index_list) > 1:
                    sub_layer_parent = getattr(
                        sub_layer_parent,
                        sub_layer_name)[sub_layer_index_list[0]]
                    for sub_layer_index in sub_layer_index_list[1:-1]:
                        sub_layer_parent = sub_layer_parent[sub_layer_index]
                    sub_layer_parent[sub_layer_index_list[-1]] = new_sub_layer
                else:
                    getattr(sub_layer_parent, sub_layer_name)[
                        sub_layer_index_list[0]] = new_sub_layer
            else:
                setattr(sub_layer_parent, sub_layer_name, new_sub_layer)

            hit_layer_pattern_list.append(pattern)
        return hit_layer_pattern_list

    def stop_after(self, stop_layer_name: str) -> bool:
        """stop forward and backward after 'stop_layer_name'.

        Args:
            stop_layer_name (str): The name of layer that stop forward and backward after this layer.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        layer_list = parse_pattern_str(stop_layer_name, self)
        if not layer_list:
            return False

        parent_layer = self
        for layer_dict in layer_list:
            name, index_list = layer_dict["name"], layer_dict["index_list"]
            if not set_identity(parent_layer, name, index_list):
                msg = f"Failed to set the layers that after stop_layer_name('{stop_layer_name}') to IdentityLayer. The error layer's name is '{name}'."
                logger.warning(msg)
                return False
            parent_layer = layer_dict["layer"]

        return True

    def freeze_befor(self, layer_name: str) -> bool:
        """freeze the layer named layer_name and its previous layer.

        Args:
            layer_name (str): The name of layer that would be freezed.

        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        def stop_grad(layer, pattern):
            class StopGradLayer(nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.layer = layer

                def forward(self, x):
                    x = self.layer(x)
                    x.stop_gradient = True
                    return x

            new_layer = StopGradLayer()
            return new_layer

        res = self.upgrade_sublayer(layer_name, stop_grad)
        if len(res) == 0:
            msg = "Failed to stop the gradient befor the layer named '{layer_name}'"
            logger.warning(msg)
            return False
        return True

    def update_res(
            self,
            return_patterns: Union[str, List[str]]) -> Dict[str, nn.Layer]:
        """update the result(s) to be returned.

        Args:
            return_patterns (Union[str, List[str]]): The name of layer to return output.

        Returns:
            Dict[str, nn.Layer]: The pattern(str) and corresponding layer(nn.Layer) that have been set successfully.
        """

        # clear res_dict that could have been set
        self.res_dict = {}

        class Handler(object):
            def __init__(self, res_dict):
                # res_dict is a reference
                self.res_dict = res_dict

            def __call__(self, layer, pattern):
                layer.res_dict = self.res_dict
                layer.res_name = pattern
                if hasattr(layer, "hook_remove_helper"):
                    layer.hook_remove_helper.remove()
                layer.hook_remove_helper = layer.register_forward_post_hook(
                    save_sub_res_hook)
                return layer

        handle_func = Handler(self.res_dict)

        hit_layer_pattern_list = self.upgrade_sublayer(
            return_patterns, handle_func=handle_func)

        if hasattr(self, "hook_remove_helper"):
            self.hook_remove_helper.remove()
        self.hook_remove_helper = self.register_forward_post_hook(
            self._return_dict_hook)

        return hit_layer_pattern_list


def save_sub_res_hook(layer, input, output):
    layer.res_dict[layer.res_name] = output


def set_identity(parent_layer: nn.Layer,
                 layer_name: str,
                 layer_index_list: str=None) -> bool:
    """set the layer specified by layer_name and layer_index_list to Indentity.

    Args:
        parent_layer (nn.Layer): The parent layer of target layer specified by layer_name and layer_index_list.
        layer_name (str): The name of target layer to be set to Indentity.
        layer_index_list (str, optional): The index of target layer to be set to Indentity in parent_layer. Defaults to None.

    Returns:
        bool: True if successfully, False otherwise.
    """

    stop_after = False
    for sub_layer_name in parent_layer._sub_layers:
        if stop_after:
            parent_layer._sub_layers[sub_layer_name] = Identity()
            continue
        if sub_layer_name == layer_name:
            stop_after = True

    if layer_index_list and stop_after:
        layer_container = parent_layer._sub_layers[layer_name]
        for num, layer_index in enumerate(layer_index_list):
            stop_after = False
            for i in range(num):
                layer_container = layer_container[layer_index_list[i]]
            for sub_layer_index in layer_container._sub_layers:
                if stop_after:
                    parent_layer._sub_layers[layer_name][
                        sub_layer_index] = Identity()
                    continue
                if layer_index == sub_layer_index:
                    stop_after = True

    return stop_after


def parse_pattern_str(pattern: str, parent_layer: nn.Layer) -> Union[
        None, List[Dict[str, Union[nn.Layer, str, None]]]]:
    """parse the string type pattern.

    Args:
        pattern (str): The pattern to discribe layer.
        parent_layer (nn.Layer): The root layer relative to the pattern.

    Returns:
        Union[None, List[Dict[str, Union[nn.Layer, str, None]]]]: None if failed. If successfully, the members are layers parsed in order:
                                                                [
                                                                    {"layer": first layer, "name": first layer's name parsed, "index": first layer's index parsed if exist},
                                                                    {"layer": second layer, "name": second layer's name parsed, "index": second layer's index parsed if exist},
                                                                    ...
                                                                ]
    """

    pattern_list = pattern.split(".")
    if not pattern_list:
        msg = f"The pattern('{pattern}') is illegal. Please check and retry."
        logger.warning(msg)
        return None

    layer_list = []
    while len(pattern_list) > 0:
        if '[' in pattern_list[0]:
            target_layer_name = pattern_list[0].split('[')[0]
            target_layer_index_list = list(
                index.split(']')[0]
                for index in pattern_list[0].split('[')[1:])
        else:
            target_layer_name = pattern_list[0]
            target_layer_index_list = None

        target_layer = getattr(parent_layer, target_layer_name, None)

        if target_layer is None:
            msg = f"Not found layer named('{target_layer_name}') specifed in pattern('{pattern}')."
            logger.warning(msg)
            return None

        if target_layer_index_list:
            for target_layer_index in target_layer_index_list:
                if int(target_layer_index) < 0 or int(
                        target_layer_index) >= len(target_layer):
                    msg = f"Not found layer by index('{target_layer_index}') specifed in pattern('{pattern}'). The index should < {len(target_layer)} and > 0."
                    logger.warning(msg)
                    return None
                target_layer = target_layer[target_layer_index]

        layer_list.append({
            "layer": target_layer,
            "name": target_layer_name,
            "index_list": target_layer_index_list
        })

        pattern_list = pattern_list[1:]
        parent_layer = target_layer

    return layer_list

 
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Tuple

class GeoPriorGen(nn.Layer):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** paddle.linspace(0, 1, embed_dim // num_heads // 2))
        angle = paddle.unsqueeze(angle, -1)
        angle = paddle.tile(angle, repeat_times=[1, 2])
        angle = paddle.flatten(angle)

        self.weight = self.create_parameter(
            shape=[2, 1, 1, 1],
            default_initializer=nn.initializer.Constant(1.0),
            is_bias=False,
        )

        heads_range_tensor = paddle.to_tensor(heads_range, dtype='float32') if not isinstance(heads_range, paddle.Tensor) else heads_range
        num_heads_tensor = paddle.to_tensor(num_heads, dtype='float32') if not isinstance(num_heads, paddle.Tensor) else num_heads

        decay = paddle.log(
            1 - 2 ** (-initial_value - heads_range_tensor * paddle.arange(num_heads, dtype='float32') / num_heads_tensor)
        )

        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        B, _, H_grid, W_grid = depth_grid.shape
        grid_d = paddle.reshape(depth_grid, [B, H_grid * W_grid, 1])
        mask_d = grid_d.unsqueeze(3) - grid_d.unsqueeze(2)  # (B, HW, 1, HW)
        mask_d = paddle.sum(paddle.abs(mask_d), axis=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return mask_d

    def generate_pos_decay(self, H: int, W: int):
        index_h = paddle.arange(H, dtype=self.decay.dtype)
        index_h = paddle.to_tensor(index_h, place=self.decay.place)

        index_w = paddle.arange(W, dtype=self.decay.dtype)
        index_w = paddle.to_tensor(index_w, place=self.decay.place)

        grid = paddle.meshgrid(index_h, index_w)
        grid = paddle.stack(grid, axis=-1)
        grid = paddle.reshape(grid, [H * W, 2])
        mask = grid.unsqueeze(1) - grid.unsqueeze(0)
        mask = paddle.sum(paddle.abs(mask), axis=-1)
        mask = mask * self.decay.unsqueeze(-1).unsqueeze(-1)
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        mask = depth_grid.unsqueeze(-1) - depth_grid.unsqueeze(-2)
        mask = paddle.abs(mask)
        mask = mask * self.decay.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # 断言维度对应H和W，Paddle的形状这里可以手动确认
        return mask

    def generate_1d_decay(self, l: int):
        index = paddle.arange(l, dtype=self.decay.dtype)
        index = paddle.to_tensor(index, place=self.decay.place)

        mask = index.unsqueeze(-1) - index.unsqueeze(0)
        mask = paddle.abs(mask)
        mask = mask * self.decay.unsqueeze(-1).unsqueeze(-1)
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        H, W = HW_tuple
        depth_map = F.interpolate(depth_map, size=[H, W], mode="bilinear", align_corners=False)

        if split_or_not:
            index = paddle.arange(H * W, dtype=self.decay.dtype)
            index = paddle.to_tensor(index, place=self.decay.place)

            sin = paddle.sin(index.unsqueeze(-1) * self.angle.unsqueeze(0))
            sin = paddle.reshape(sin, [H, W, -1])
            cos = paddle.cos(index.unsqueeze(-1) * self.angle.unsqueeze(0))
            cos = paddle.reshape(cos, [H, W, -1])

            mask_d_h = self.generate_1d_depth_decay(H, W, paddle.transpose(depth_map, perm=[0,1,3,2]))
            mask_d_w = self.generate_1d_depth_decay(W, H, depth_map)

            mask_h = self.generate_1d_decay(H)
            mask_w = self.generate_1d_decay(W)

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            geo_prior = ((sin, cos), (mask_h, mask_w))

        else:
            index = paddle.arange(H * W, dtype=self.decay.dtype)
            index = paddle.to_tensor(index, place=self.decay.place)

            sin = paddle.sin(index.unsqueeze(-1) * self.angle.unsqueeze(0))
            sin = paddle.reshape(sin, [H, W, -1])
            cos = paddle.cos(index.unsqueeze(-1) * self.angle.unsqueeze(0))
            cos = paddle.reshape(cos, [H, W, -1])

            mask = self.generate_pos_decay(H, W)
            mask_d = self.generate_depth_decay(H, W, depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d

            geo_prior = ((sin, cos), mask)

        return geo_prior

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DWConv2d(nn.Layer):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x):
        """
        input (b h w c)
        """
        x = paddle.transpose(x, [0, 3, 1, 2])
        x = self.dwconv(x)
        x = paddle.transpose(x, [0, 2, 3, 1])
        return x


import paddle

def angle_transform(x, sin, cos):
    """
    x: [B, heads, H, W, dim]
    sin, cos: [H, W, dim_sin] (没有batch维，dim_sin需和x最后一维dim兼容)
    """
    B, heads, H, W, dim = x.shape
    # 假设 sin, cos 是 [H, W, dim_sin], 需要和 dim 对齐，如果不一样，可能要pad或slice
    if sin.shape[-1] != dim:
        # 这里简单截断或填充0到dim大小
        if sin.shape[-1] > dim:
            sin = sin[:, :, :dim]
            cos = cos[:, :, :dim]
        else:
            pad_len = dim - sin.shape[-1]
            sin = paddle.concat([sin, paddle.zeros([H, W, pad_len], dtype=sin.dtype)], axis=-1)
            cos = paddle.concat([cos, paddle.zeros([H, W, pad_len], dtype=cos.dtype)], axis=-1)

    # 直接unsqueeze batch和heads维度，广播
    sin = sin.unsqueeze(0).unsqueeze(0).expand([B, heads, H, W, dim])
    cos = cos.unsqueeze(0).unsqueeze(0).expand([B, heads, H, W, dim])

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    stacked = paddle.stack([-x2, x1], axis=-1)
    stacked = paddle.flatten(stacked, start_axis=-2)

    return x * cos + stacked * sin





class CrossGSA(nn.Layer):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor)

        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim)
        
        self.layer_norm = nn.LayerNorm(self.embed_dim, epsilon=1e-6)
        self.reset_parameters()

    def forward(self, x: paddle.Tensor, y: paddle.Tensor = None, rel_pos=None):
        """
        x: image feature [b h w c]
        y: depth feature [b h w c]
        rel_pos: (sin, cos), mask from GeoPriorGen
        """
        bsz, h, w, _ = x.shape
        q = self.q_proj(x)
        if y is not None:
            k = self.k_proj(y)
            v = self.v_proj(y)
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling

        qr = paddle.reshape(q, [bsz, h, w, self.num_heads, -1])
        qr = paddle.transpose(qr, [0, 3, 1, 2, 4])
        kr = paddle.reshape(k, [bsz, h, w, self.num_heads, -1])
        kr = paddle.transpose(kr, [0, 3, 1, 2, 4])
        vr = paddle.reshape(v, [bsz, h, w, self.num_heads, -1])
        vr = paddle.transpose(vr, [0, 3, 1, 2, 4])
        
        if rel_pos is not None:
            
            sin, cos = rel_pos[0]
            qr = angle_transform(qr, sin, cos)
            kr = angle_transform(kr, sin, cos)

        qr = paddle.flatten(qr, start_axis=2, stop_axis=3)  # flatten spatial dims
        kr = paddle.flatten(kr, start_axis=2, stop_axis=3)
        vr = paddle.flatten(vr, start_axis=2, stop_axis=3)

        qk_mat = paddle.matmul(qr, paddle.transpose(kr, perm=[0,1,3,2]))
        #print("qk_mat shape:", qk_mat.shape)

        if rel_pos is not None:
            _, mask = rel_pos
            qk_mat = qk_mat + mask
            #print("mask shape:", mask.shape)

        qk_mat = F.softmax(qk_mat, axis=-1)
        output = paddle.matmul(qk_mat, vr)

        output = paddle.transpose(output, perm=[0, 2, 1, 3])
        output = paddle.reshape(output, [bsz, h, w, -1])
        output = output + lepe
        output = x + self.out_proj(output)

        output = output + self.layer_norm(output)

        return output
    
    def reset_parameters(self):
        nn.initializer.XavierNormal()(self.q_proj.weight)
        nn.initializer.XavierNormal()(self.k_proj.weight)
        nn.initializer.XavierNormal()(self.v_proj.weight)
        nn.initializer.XavierNormal()(self.out_proj.weight)
        
        
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class EMAA(nn.Layer):
    def __init__(self, channels, factor=16):
        super(EMAA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(axis=-1)

        self.agp = nn.AdaptiveAvgPool2D((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2D((1, None))

        self.gn = nn.GroupNorm(num_groups=channels // self.groups, num_channels=channels // self.groups)
        self.conv1x1 = nn.Conv2D(
            in_channels=channels // self.groups,
            out_channels=channels // self.groups,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv3x3 = nn.Conv2D(
            in_channels=channels // self.groups,
            out_channels=channels // self.groups,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        b, c, h, w = x.shape
        group_x = x.reshape([b * self.groups, -1, h, w])  # [b*g, c//g, h, w]

        x_h = self.pool_h(group_x)  # [b*g, c//g, h, 1]
        x_w = self.pool_w(group_x).transpose([0, 1, 3, 2])  # [b*g, c//g, 1, w]

        hw = self.conv1x1(paddle.concat([x_h, x_w], axis=2))  # [b*g, c//g, h+w, 1]
        x_h, x_w = paddle.split(hw, num_or_sections=[h, w], axis=2)
        x_w = x_w.transpose([0, 1, 3, 2])  # [b*g, c//g, 1, w]

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape([b * self.groups, -1, 1]).transpose([0, 2, 1]))  # [b*g, 1, c//g]
        x12 = x2.reshape([b * self.groups, c // self.groups, -1])  # [b*g, c//g, h*w]

        x21 = self.softmax(self.agp(x2).reshape([b * self.groups, -1, 1]).transpose([0, 2, 1]))  # [b*g, 1, c//g]
        x22 = x1.reshape([b * self.groups, c // self.groups, -1])  # [b*g, c//g, h*w]

        weights = paddle.matmul(x11, x12) + paddle.matmul(x21, x22)  # [b*g, 1, h*w]
        weights = weights.reshape([b * self.groups, 1, h, w])

        out = group_x * weights.sigmoid()
        return out.reshape([b, c, h, w])

        




