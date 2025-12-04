import random
import numpy as np
import logging
import uuid

# 移除固定随机种子，改为动态设置
# 随机种子将在需要时通过函数参数或环境变量设置

class RandomSeedManager:
    """随机种子管理器，用于统一管理variation模块的随机种子"""
    
    _instance = None
    _seed = None
    _is_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RandomSeedManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def set_seed(cls, seed: int = None):
        """
        设置随机种子
        Args:
            seed: 随机种子值，如果为None则使用时间戳生成
        """
        if seed is None:
            import time
            seed = int(time.time()) % 1000000
        
        cls._seed = seed
        cls._is_initialized = True
        
        # 设置所有相关的随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        logging.info(f"🎲 Variation模块随机种子已设置为: {seed}")
        return seed
    
    @classmethod
    def get_seed(cls):
        """获取当前随机种子"""
        if not cls._is_initialized:
            return cls.set_seed()  # 如果未初始化，自动设置
        return cls._seed
    
    @classmethod
    def reset_seed(cls):
        """重置随机种子（使用新的时间戳）"""
        return cls.set_seed()
    
    @classmethod
    def is_initialized(cls):
        """检查是否已初始化随机种子"""
        return cls._is_initialized

class VariationConfig:
    """变异配置，定义变异结构和互斥关系
    
    新的变异类别结构：
    V1 (System Message Mutation): 系统消息变异
    V2 (User/Assistant Message Mutation): 用户/助手消息变异  
    V3 (Role Marker Mutation): 角色标记变异 (合并原V3+V4)
    V4_1 (Delimiter Mutation 1): 分隔符变异1 (原V5)
    V4_2 (Delimiter Mutation 2): 分隔符变异2 (原V6_1-V6_3)
    V4_3 (Delimiter Mutation 3): 分隔符变异3 (原V6_4-V6_6)
    V4_4 (Delimiter Mutation 4): 分隔符变异4 (原V6_7-V6_9)
    V5 (Remove generation hint): 移除生成提示 (原V7+V8)
    """
    def __init__(self):
        # 新的变异映射：从新变异名到原变异名
        self.new_to_old_mapping = {
            # V1 保持不变
            "V1_1": "V1_1", "V1_2": "V1_2", "V1_3": "V1_3", "V1_4": "V1_4", "V1_5": "V1_5", "V1_6": "V1_6",
            # V2 保持不变  
            "V2_1": "V2_1", "V2_2": "V2_2", "V2_3": "V2_3", "V2_4": "V2_4", "V2_5": "V2_5",
            # V3 合并原V3和V4
            "V3_1": "V3_1", "V3_2": "V3_2", "V3_3": "V3_3", "V3_4": "V3_4",
            "V3_5": "V4_1", "V3_6": "V4_2", "V3_7": "V4_3", "V3_8": "V4_4", "V3_9": "V4_5", "V3_10": "V4_6",
            "V3_11": "V4_7", "V3_12": "V4_8", "V3_13": "V4_9", "V3_14": "V4_10", "V3_15": "V4_11", "V3_16": "V4_12",
            "V3_17": "V4_13", "V3_18": "V4_14", "V3_19": "V4_15", "V3_20": "V4_16",
            # V4_1 对应原V5
            "V4_1_1": "V5_1", "V4_1_2": "V5_2", "V4_1_3": "V5_3", "V4_1_4": "V5_4", "V4_1_5": "V5_5", "V4_1_6": "V5_6", "V4_1_7": "V5_7",
            # V4_2 对应原V6_1-V6_3
            "V4_2_1": "V6_1", "V4_2_2": "V6_2", "V4_2_3": "V6_3",
            # V4_3 对应原V6_4-V6_6
            "V4_3_1": "V6_4", "V4_3_2": "V6_5", "V4_3_3": "V6_6",
            # V4_4 对应原V6_7-V6_9
            "V4_4_1": "V6_7", "V4_4_2": "V6_8", "V4_4_3": "V6_9",
            # V5 合并原V7和V8
            "V5_1": "V7_1", "V5_2": "V7_2", "V5_3": "V7_3",
            "V5_4": "V8_1", "V5_5": "V8_2", "V5_6": "V8_3", "V5_7": "V8_4", "V5_8": "V8_5",
            "V5_9": "V8_6", "V5_10": "V8_7", "V5_11": "V8_8", "V5_12": "V8_9", "V5_13": "V8_10"
        }
        
        # 反向映射：从原变异名到新变异名
        self.old_to_new_mapping = {v: k for k, v in self.new_to_old_mapping.items()}
        
        # 新的冲突组定义
        self.conflict_groups = {
            # 大类之间不互斥，只有各大类内部的小类互斥
            "V1": ["V1_2", "V1_3", "V1_4", "V1_5", "V1_6"],  # V1_1是默认，不参与互斥
            "V2": ["V2_1", "V2_2", "V2_3", "V2_4", "V2_5"],
            "V3": ["V3_1", "V3_2", "V3_3", "V3_4", "V3_5", "V3_6", "V3_7", "V3_8", "V3_9", "V3_10", 
                   "V3_11", "V3_12", "V3_13", "V3_14", "V3_15", "V3_16", "V3_17", "V3_18", "V3_19", "V3_20"],
            "V4_1": ["V4_1_1", "V4_1_2", "V4_1_3", "V4_1_4", "V4_1_5", "V4_1_6", "V4_1_7"],
            "V4_2": ["V4_2_1", "V4_2_2", "V4_2_3"],
            "V4_3": ["V4_3_1", "V4_3_2", "V4_3_3"],
            "V4_4": ["V4_4_1", "V4_4_2", "V4_4_3"],
            "V5": ["V5_1", "V5_2", "V5_3", "V5_4", "V5_5", "V5_6", "V5_7", "V5_8", "V5_9", "V5_10", "V5_11", "V5_12", "V5_13"],
            
            # V4的特殊互斥关系：V4_1与V4_2,V4_3,V4_4互斥，但V4_2,V4_3,V4_4之间不互斥
            "V4_DELIMITER_CONFLICT": ["V4_1_1", "V4_1_2", "V4_1_3", "V4_1_4", "V4_1_5", "V4_1_6", "V4_1_7",
                                     "V4_2_1", "V4_2_2", "V4_2_3", "V4_3_1", "V4_3_2", "V4_3_3", "V4_4_1", "V4_4_2", "V4_4_3"]
        }
        
        # 新的类别配置
        self.class_config = {
            "V1": {0: [], 1: ["V1_1"], 2: ["V1_2"], 3: ["V1_3"], 4: ["V1_4"], 5: ["V1_5"], 6: ["V1_6"]},
            "V2": {0: [], 1: ["V2_1"], 2: ["V2_2"], 3: ["V2_3"], 4: ["V2_4"], 5: ["V2_5"]},
            "V3": {0: [], 1: ["V3_1"], 2: ["V3_2"], 3: ["V3_3"], 4: ["V3_4"], 5: ["V3_5"], 6: ["V3_6"], 7: ["V3_7"], 8: ["V3_8"], 
                   9: ["V3_9"], 10: ["V3_10"], 11: ["V3_11"], 12: ["V3_12"], 13: ["V3_13"], 14: ["V3_14"], 15: ["V3_15"], 
                   16: ["V3_16"], 17: ["V3_17"], 18: ["V3_18"], 19: ["V3_19"], 20: ["V3_20"]},
            "V4_1": {0: [], 1: ["V4_1_1"], 2: ["V4_1_2"], 3: ["V4_1_3"], 4: ["V4_1_4"], 5: ["V4_1_5"], 6: ["V4_1_6"], 7: ["V4_1_7"]},
            "V4_2": {0: [], 1: ["V4_2_1"], 2: ["V4_2_2"], 3: ["V4_2_3"]},
            "V4_3": {0: [], 1: ["V4_3_1"], 2: ["V4_3_2"], 3: ["V4_3_3"]},
            "V4_4": {0: [], 1: ["V4_4_1"], 2: ["V4_4_2"], 3: ["V4_4_3"]},
            "V5": {0: [], 1: ["V5_1"], 2: ["V5_2"], 3: ["V5_3"], 4: ["V5_4"], 5: ["V5_5"], 6: ["V5_6"], 7: ["V5_7"], 
                   8: ["V5_8"], 9: ["V5_9"], 10: ["V5_10"], 11: ["V5_11"], 12: ["V5_12"], 13: ["V5_13"]}
        }
        # Variant to class mapping
        self.variant_to_class = {}
        for class_name, variants in self.class_config.items():
            for idx, variant_list in variants.items():
                for variant in variant_list:
                    self.variant_to_class[variant] = class_name
        # Validate conflict groups
        for group, variants in self.conflict_groups.items():
            for variant in variants:
                if variant not in self.variant_to_class:
                    logging.warning(f"变异 {variant} 在 conflict_groups 中，但在 class_config 中未定义")

class VariationProbabilityManager:
    """管理大类和小类变异概率"""
    def __init__(self, config: VariationConfig):
        self.config = config
        
        # 确保随机种子已初始化
        if not RandomSeedManager.is_initialized():
            RandomSeedManager.set_seed()
            logging.debug("VariationProbabilityManager: 自动初始化随机种子")
        # Initialize 8 major class probabilities (新的分组结构)
        self.major_probs = {
            "V1": 1/8, "V2": 1/8, "V3": 1/8, "V4_1": 1/8, "V4_2": 1/8, "V4_3": 1/8, "V4_4": 1/8, "V5": 1/8
        }
        self.minor_probs = {}
        for class_name in config.class_config:
            variants = [v for idx, v in config.class_config[class_name].items() if idx != 0]
            # 直接使用新的变异名，不需要映射
            class_variants = {}
            for v in variants:
                variant_name = v[0]  # 获取变异名
                class_variants[variant_name] = 1/len(variants)
            self.minor_probs[class_name] = class_variants
        logging.debug("初始化变异概率: 大类=%s, 小类=%s", self.major_probs, self.minor_probs)

    def initialize_probabilities(self, scores: dict):
        """根据初始实验分数初始化概率"""
        logging.info("开始初始化变异概率")
        # Compute major class average scores
        major_scores = {class_name: 0 for class_name in self.major_probs}
        major_counts = {class_name: 0 for class_name in self.major_probs}
        for variant, score in scores.items():
            class_name = self.config.variant_to_class.get(variant, None)
            if class_name:
                major_scores[class_name] += score
                major_counts[class_name] += 1
                logging.debug("变异 %s (大类 %s) 分数: %.4f", variant, class_name, score)
        # Set major class probabilities
        raw_probs = {}
        total_raw = 0
        for class_name in major_scores:
            avg_score = major_scores[class_name] / max(1, major_counts[class_name])
            raw_probs[class_name] = (avg_score + 0.1) ** 2
            total_raw += raw_probs[class_name]
        # Normalize and cap max probability
        min_prob = min(raw_probs.values()) / total_raw
        max_allowed = 3 * min_prob
        total_prob = 0
        for class_name in raw_probs:
            prob = raw_probs[class_name] / total_raw
            self.major_probs[class_name] = round(min(prob, max_allowed), 2)
            total_prob += self.major_probs[class_name]
        # Normalize again
        for class_name in self.major_probs:
            self.major_probs[class_name] = round(self.major_probs[class_name] / total_prob, 2)
        logging.info("大类概率: %s", self.major_probs)
        # Set minor class probabilities
        for class_name in self.minor_probs:
            variants = self.minor_probs[class_name]
            num_variants = len(variants)
            for variant in variants:
                variants[variant] = round(1 / num_variants, 2)
            logging.info("小类概率 (%s): %s", class_name, variants)

    def update_probabilities(self, parent_variants: list, new_variant: str, old_score: float, new_score: float, seed_pool_scores: list = None, learning_rate: float = 2):
        """根据变异效果动态调整概率"""
        logging.debug("更新概率: 父变异=%s, 新变异=%s, 旧分数=%.4f, 新分数=%.4f", parent_variants, new_variant, old_score, new_score)
        # Compute dynamic baseline score
        BASELINE_SCORE = min(seed_pool_scores) if seed_pool_scores else 0.5840
        logging.debug("动态基准分数: %.4f", BASELINE_SCORE)
    
        # Compute change ratios
        parent_change = (new_score - old_score) / max(old_score, 1e-6)
        baseline_change = (new_score - BASELINE_SCORE) / max(BASELINE_SCORE, 1e-6) if new_score > old_score else 0
        if new_score > old_score:
            adjustment = 1 + learning_rate * baseline_change + learning_rate * parent_change
            adjustment = min(max(adjustment, 1.0), 2.0)
            logging.debug("分数提升，调整因子=%.2f", adjustment)
            variants_to_update = parent_variants + [new_variant] if new_score > BASELINE_SCORE else [new_variant]
        else:
            adjustment = 1 - learning_rate * abs(parent_change)
            adjustment = min(max(adjustment, 0.5), 1.0)
            logging.debug("分数下降，调整因子=%.2f", adjustment)
            variants_to_update = [new_variant]

        # Update major and minor probabilities
        updated_classes = set()
        for variant in variants_to_update:
            class_name = self.config.variant_to_class.get(variant, None)
            if class_name and class_name not in updated_classes:
                # Update major probabilities
                total_prob = sum(self.major_probs.values())
                self.major_probs[class_name] *= adjustment
                new_total = sum(self.major_probs.values())
                for k in self.major_probs:
                    self.major_probs[k] = round(self.major_probs[k] * total_prob / new_total, 2)
                # Update minor probabilities
                variants = self.minor_probs[class_name]
                # 直接使用新的变异名
                if variant in variants:
                    total_prob = sum(variants.values())
                    variants[variant] *= adjustment
                    new_total = sum(variants.values())
                    for k in variants:
                        variants[k] = round(variants[k] * total_prob / new_total, 2)
                else:
                    logging.warning(f"变异 {variant} 在 minor_probs[{class_name}] 中未找到")
                updated_classes.add(class_name)

    def select_variation(self, seed_classes: list = None) -> str:
        """根据概率选择一个变异"""
        classes = list(self.major_probs.keys())
        probs = [self.major_probs[c] for c in classes]
        if seed_classes:
            classes = [c for c in classes if c in seed_classes]
            probs = [self.major_probs[c] for c in classes]
            if not classes:
                logging.warning("没有可用的种子大类: %s", seed_classes)
                return None
        class_name = np.random.choice(classes, p=np.array(probs) / sum(probs))
        variants = list(self.minor_probs[class_name].keys())
        probs = [self.minor_probs[class_name][v] for v in variants]
        selected_variant = np.random.choice(variants, p=np.array(probs) / sum(probs))
        
        logging.debug("选择变异: 大类=%s, 选中变异=%s", class_name, selected_variant)
        return selected_variant
    
    def set_random_seed(self, seed: int = None):
        """
        为概率管理器设置随机种子
        Args:
            seed: 随机种子值，如果为None则使用时间戳生成
        Returns:
            实际使用的随机种子值
        """
        actual_seed = RandomSeedManager.set_seed(seed)
        logging.info(f"🎲 VariationProbabilityManager: 随机种子已设置为 {actual_seed}")
        return actual_seed
    
    def get_current_seed(self):
        """获取当前使用的随机种子"""
        return RandomSeedManager.get_seed()

def check_v4_conflicts(variants, config):
    """检查V4的特殊冲突关系：V4_1与V4_2,V4_3,V4_4互斥"""
    v4_1_variants = [v for v in variants if v in config.conflict_groups.get("V4_1", [])]
    v4_234_variants = []
    for v in variants:
        if (v in config.conflict_groups.get("V4_2", []) or 
            v in config.conflict_groups.get("V4_3", []) or 
            v in config.conflict_groups.get("V4_4", [])):
            v4_234_variants.append(v)
    
    # 如果V4_1和V4_2/V4_3/V4_4同时存在，则有冲突
    return len(v4_1_variants) > 0 and len(v4_234_variants) > 0

def generate_random_template(max_variations, config, prob_manager: VariationProbabilityManager, seed_classes=None, parent_variants=None):
    # 确保随机种子已初始化
    if not RandomSeedManager.is_initialized():
        RandomSeedManager.set_seed()
        logging.debug("generate_random_template: 自动初始化随机种子")
    
    selected_variants = []
    group_variants = {group: [] for group in config.conflict_groups}
    selected_subgroups = set()
    parent_variants = parent_variants or []

    variant_to_subgroup = {}
    for group, variants in config.conflict_groups.items():
        for variant in variants:
            variant_to_subgroup[variant] = config.variant_to_class[variant]

    for variant in parent_variants:
        for group, variants in config.conflict_groups.items():
            if variant in variants:
                group_variants[group].append(variant)
        if variant in variant_to_subgroup:
            selected_subgroups.add(variant_to_subgroup[variant])

    while len(selected_variants) < max_variations:
        variant = prob_manager.select_variation(seed_classes)
        if not variant:
            break
        subgroup = variant_to_subgroup.get(variant, config.variant_to_class[variant])
        if subgroup in selected_subgroups:
            continue
        can_add = True
        conflicting_groups = []
        for group, variants in config.conflict_groups.items():
            if variant in variants:
                if group_variants[group]:
                    can_add = False
                    conflicting_groups.append(group)
                else:
                    group_variants[group].append(variant)
        if can_add and variant not in selected_variants:
            selected_variants.append(variant)
            selected_subgroups.add(subgroup)

    all_variants = parent_variants + selected_variants
    class_counts = {}
    for variant in all_variants:
        class_name = config.variant_to_class[variant]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    # 新的结构中，各大类之间不冲突，只需检查大类内部冲突
    conflict_classes = [c for c, count in class_counts.items() if count > 1]
    conflict_groups = []
    for group, group_variants in config.conflict_groups.items():
        group_count = sum(1 for variant in all_variants if variant in group_variants)
        if group_count > 1:
            conflict_groups.append((group, [v for v in all_variants if v in group_variants]))
    
    # 检查V4的特殊冲突
    v4_conflict = check_v4_conflicts(all_variants, config)
    
    if conflict_classes or conflict_groups or v4_conflict:
        return {class_name: [] for class_name in config.class_config}, []

    def variant_to_class_index(variant, class_name):
        """将变异名转换为对应类别的索引"""
        # 处理新的命名规则
        parts = variant.split("_")
        if len(parts) == 2:
            # V1_1, V2_1 等格式
            variant_idx = int(parts[1])
            return variant_idx
        elif len(parts) == 3:
            # V4_1_1, V4_2_1 等格式  
            variant_idx = int(parts[2])
            return variant_idx
        else:
            # 其他情况，尝试从最后一个部分提取索引
            try:
                variant_idx = int(parts[-1])
                return variant_idx
            except ValueError:
                logging.warning(f"无法解析变异索引: {variant}")
                return 1  # 默认返回1

    template = {class_name: [] for class_name in config.class_config}
    for variant in selected_variants:
        class_name = config.variant_to_class[variant]
        variant_idx = variant_to_class_index(variant, class_name)
        template[class_name].append(variant_idx)

    return template, selected_variants

class RandomMutationGenerator:
    """完全随机变异组合生成器"""
    def __init__(self, config: 'VariationConfig'):
        self.config = config
        self.used_combinations = set()  # 记录已使用的变异组合
        
        # 确保随机种子已初始化
        if not RandomSeedManager.is_initialized():
            RandomSeedManager.set_seed()
            logging.debug("RandomMutationGenerator: 自动初始化随机种子")
        
    def generate_random_combination(self, max_variations: int) -> tuple:
        """
        生成完全随机的变异组合
        改进逻辑：
        1. 使用指定的大类选择概率：
           V1: 1/8, V2: 1/8, V3: 1/8, V4: 1/8, V5: 1/16, 
           V6_BOS: 1/16, V6_BOT: 1/16, V6_ROLE: 1/16, V7: 1/8, V8: 1/8
        2. 大类内所有小类概率相等
        
        Args:
            max_variations: 最大变异叠加数
        Returns:
            (template_dict, variants_list) 或 (None, None) 如果无法生成有效组合
        """
        max_attempts = 1000  # 最大尝试次数，避免无限循环
        
        # 定义大类选择概率（新的分组结构）
        class_probabilities = {
            "V1": 1/8,
            "V2": 1/8, 
            "V3": 1/8,
            "V4_1": 1/8,
            "V4_2": 1/8,
            "V4_3": 1/8,
            "V4_4": 1/8,
            "V5": 1/8
        }
        
        for attempt in range(max_attempts):
            # 获取所有可用的大类
            available_classes = list(self.config.class_config.keys())
            
            if not available_classes:
                logging.warning("🎲 随机变异模式：没有可用的大类")
                return None, None
            
            # 随机选择变异数量（1到max_variations之间）
            num_variations = random.randint(1, max_variations)
            
            # 新的选择逻辑：先选大类，再选小类
            selected_variants = []
            selected_classes = set()
            
            # 用于跟踪冲突组中已选择的变异
            conflict_group_selections = {group: [] for group in self.config.conflict_groups}
            
            for _ in range(num_variations):
                # 第一步：获取可选择的大类（排除已选择的和冲突的）
                available_classes_for_selection = []
                available_probabilities = []
                
                for class_name in available_classes:
                    if class_name in selected_classes:
                        continue
                    
                    # 检查该大类是否有可用的变异（排除冲突）
                    class_has_available_variants = False
                    for idx, variant_list in self.config.class_config[class_name].items():
                        if idx == 0 or not variant_list:  # 跳过无变异选项
                            continue
                        
                        variant = variant_list[0]  # 获取变异名
                        
                        # 检查互斥关系
                        can_select = True
                        for group, group_variants in self.config.conflict_groups.items():
                            if variant in group_variants:
                                # 检查该冲突组是否已有选择
                                if conflict_group_selections[group]:
                                    can_select = False
                                    break
                        
                        # 检查V4的特殊冲突关系
                        if can_select:
                            # 构造临时变异列表进行V4冲突检查
                            temp_variants = selected_variants + [variant]
                            if check_v4_conflicts(temp_variants, self.config):
                                can_select = False
                        
                        if can_select:
                            class_has_available_variants = True
                            break
                    
                    if class_has_available_variants:
                        available_classes_for_selection.append(class_name)
                        # 获取该大类的概率，如果没有定义则使用默认值
                        prob = class_probabilities.get(class_name, 1/16)
                        available_probabilities.append(prob)
                
                if not available_classes_for_selection:
                    break
                
                # 第二步：根据指定概率选择一个大类
                # 归一化概率
                total_prob = sum(available_probabilities)
                normalized_probs = [p / total_prob for p in available_probabilities]
                
                # 使用概率权重选择大类
                selected_class = random.choices(available_classes_for_selection, weights=normalized_probs)[0]
                logging.debug(f"🎲 按概率选择大类: {selected_class} (概率: {class_probabilities.get(selected_class, 1/16):.4f})")
                
                # 第三步：在选中的大类内，等概率选择一个小类变异
                available_variants_in_class = []
                for idx, variant_list in self.config.class_config[selected_class].items():
                    if idx == 0 or not variant_list:  # 跳过无变异选项
                        continue
                    
                    variant = variant_list[0]  # 获取变异名
                    
                    # 检查互斥关系
                    can_select = True
                    for group, group_variants in self.config.conflict_groups.items():
                        if variant in group_variants:
                            # 检查该冲突组是否已有选择
                            if conflict_group_selections[group]:
                                can_select = False
                                break
                    
                    # 检查V4的特殊冲突关系
                    if can_select:
                        # 构造临时变异列表进行V4冲突检查
                        temp_variants = selected_variants + [variant]
                        if check_v4_conflicts(temp_variants, self.config):
                            can_select = False
                    
                    if can_select:
                        available_variants_in_class.append(variant)
                
                if not available_variants_in_class:
                    continue
                
                # 在大类内等概率随机选择一个小类变异
                selected_variant = random.choice(available_variants_in_class)
                logging.debug(f"🎲 在大类 {selected_class} 中随机选择小类: {selected_variant}")
                
                selected_variants.append(selected_variant)
                selected_classes.add(selected_class)
                
                # 更新冲突组选择
                for group, group_variants in self.config.conflict_groups.items():
                    if selected_variant in group_variants:
                        conflict_group_selections[group].append(selected_variant)
            
            if not selected_variants:
                continue
            
            # 生成组合签名用于去重
            combination_signature = tuple(sorted(selected_variants))
            
            # 检查是否重复
            if combination_signature in self.used_combinations:
                continue
            
            # 记录已使用的组合
            self.used_combinations.add(combination_signature)
            
            # 转换为模板格式
            template = self._variants_to_template(selected_variants)
            
            logging.info(f"🎲 随机变异模式：生成新组合 (尝试 {attempt + 1}): {selected_variants}")
            logging.debug(f"🎲 选择过程: 大类按指定概率选择 -> 大类内小类概率相等")
            logging.debug(f"🎲 大类概率: V1/V2/V3/V4/V7=1/8, V5/V6_BOS/V6_BOT/V6_ROLE/V8_PROMPT/V8_SEP=1/16")
            return template, selected_variants
        
        logging.warning(f"🎲 随机变异模式：经过 {max_attempts} 次尝试，无法生成新的随机变异组合")
        return None, None
    
    def _variants_to_template(self, variants: list) -> dict:
        """将变异列表转换为模板字典格式"""
        def variant_to_class_index(variant, class_name):
            """将变异名转换为对应类别的索引"""
            # 处理新的命名规则
            parts = variant.split("_")
            if len(parts) == 2:
                # V1_1, V2_1 等格式
                variant_idx = int(parts[1])
                return variant_idx
            elif len(parts) == 3:
                # V4_1_1, V4_2_1 等格式  
                variant_idx = int(parts[2])
                return variant_idx
            else:
                # 其他情况，尝试从最后一个部分提取索引
                try:
                    variant_idx = int(parts[-1])
                    return variant_idx
                except ValueError:
                    logging.warning(f"无法解析变异索引: {variant}")
                    return 1  # 默认返回1
        
        template = {class_name: [] for class_name in self.config.class_config}
        for variant in variants:
            class_name = self.config.variant_to_class[variant]
            variant_idx = variant_to_class_index(variant, class_name)
            template[class_name].append(variant_idx)
        
        return template
    
    def reset_used_combinations(self):
        """重置已使用的组合记录"""
        self.used_combinations.clear()
        logging.info("🎲 随机变异模式：已重置变异组合记录")
    
    def get_used_combinations_count(self) -> int:
        """获取已使用的组合数量"""
        return len(self.used_combinations)
    
    def set_random_seed(self, seed: int = None):
        """
        为当前生成器设置随机种子
        Args:
            seed: 随机种子值，如果为None则使用时间戳生成
        Returns:
            实际使用的随机种子值
        """
        actual_seed = RandomSeedManager.set_seed(seed)
        logging.info(f"🎲 RandomMutationGenerator: 随机种子已设置为 {actual_seed}")
        return actual_seed
    
    def get_current_seed(self):
        """获取当前使用的随机种子"""
        return RandomSeedManager.get_seed()
    
    def reset_with_new_seed(self):
        """使用新的随机种子重置生成器，并清空已使用的组合记录"""
        new_seed = RandomSeedManager.reset_seed()
        self.used_combinations.clear()
        logging.info(f"🎲 RandomMutationGenerator: 已使用新种子 {new_seed} 重置，清空组合记录")
        return new_seed


# 便利函数，用于外部直接调用
def set_variation_random_seed(seed: int = None):
    """
    设置variation模块的随机种子
    Args:
        seed: 随机种子值，如果为None则使用时间戳生成
    Returns:
        实际使用的随机种子值
    """
    return RandomSeedManager.set_seed(seed)

def get_variation_random_seed():
    """获取当前variation模块的随机种子"""
    return RandomSeedManager.get_seed()

def reset_variation_random_seed():
    """重置variation模块的随机种子（使用新的时间戳）"""
    return RandomSeedManager.reset_seed()

def is_variation_seed_initialized():
    """检查variation模块的随机种子是否已初始化"""
    return RandomSeedManager.is_initialized()
