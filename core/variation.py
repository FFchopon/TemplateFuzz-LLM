import random
import numpy as np
import logging
import uuid

# Fixed random seed removed; seed is set dynamically instead
# Random seed is configured via function arguments or environment variables when needed

class RandomSeedManager:
    """Random seed manager for unified seed control in the variation module."""
    
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
        Set random seed.
        Args:
            seed: Seed value; if None, generated from timestamp
        """
        if seed is None:
            import time
            seed = int(time.time()) % 1000000
        
        cls._seed = seed
        cls._is_initialized = True
        
        # Set all related random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        logging.info(f"🎲 Variation module random seed set to: {seed}")
        return seed
    
    @classmethod
    def get_seed(cls):
        """Get current random seed."""
        if not cls._is_initialized:
            return cls.set_seed()  # Auto-initialize if not yet set
        return cls._seed
    
    @classmethod
    def reset_seed(cls):
        """Reset random seed (using a new timestamp)."""
        return cls.set_seed()
    
    @classmethod
    def is_initialized(cls):
        """Check whether random seed has been initialized."""
        return cls._is_initialized

class VariationConfig:
    """Variation configuration defining structure and mutual exclusion.
    
    New variation category structure:
    V1 (System Message Mutation): System message mutation
    V2 (User/Assistant Message Mutation): User/assistant message mutation
    V3 (Role Marker Mutation): Role marker mutation (merged former V3+V4)
    V4_1 (Delimiter Mutation 1): Delimiter mutation 1 (former V5)
    V4_2 (Delimiter Mutation 2): Delimiter mutation 2 (former V6_1-V6_3)
    V4_3 (Delimiter Mutation 3): Delimiter mutation 3 (former V6_4-V6_6)
    V4_4 (Delimiter Mutation 4): Delimiter mutation 4 (former V6_7-V6_9)
    V5 (Remove generation hint): Remove generation hint (former V7+V8)
    """
    def __init__(self):
        # New variation mapping: new name -> old name
        self.new_to_old_mapping = {
            # V1 unchanged
            "V1_1": "V1_1", "V1_2": "V1_2", "V1_3": "V1_3", "V1_4": "V1_4", "V1_5": "V1_5", "V1_6": "V1_6",
            # V2 unchanged
            "V2_1": "V2_1", "V2_2": "V2_2", "V2_3": "V2_3", "V2_4": "V2_4", "V2_5": "V2_5",
            # V3 merges former V3 and V4
            "V3_1": "V3_1", "V3_2": "V3_2", "V3_3": "V3_3", "V3_4": "V3_4",
            "V3_5": "V4_1", "V3_6": "V4_2", "V3_7": "V4_3", "V3_8": "V4_4", "V3_9": "V4_5", "V3_10": "V4_6",
            "V3_11": "V4_7", "V3_12": "V4_8", "V3_13": "V4_9", "V3_14": "V4_10", "V3_15": "V4_11", "V3_16": "V4_12",
            "V3_17": "V4_13", "V3_18": "V4_14", "V3_19": "V4_15", "V3_20": "V4_16",
            # V4_1 maps to former V5
            "V4_1_1": "V5_1", "V4_1_2": "V5_2", "V4_1_3": "V5_3", "V4_1_4": "V5_4", "V4_1_5": "V5_5", "V4_1_6": "V5_6", "V4_1_7": "V5_7",
            # V4_2 maps to former V6_1-V6_3
            "V4_2_1": "V6_1", "V4_2_2": "V6_2", "V4_2_3": "V6_3",
            # V4_3 maps to former V6_4-V6_6
            "V4_3_1": "V6_4", "V4_3_2": "V6_5", "V4_3_3": "V6_6",
            # V4_4 maps to former V6_7-V6_9
            "V4_4_1": "V6_7", "V4_4_2": "V6_8", "V4_4_3": "V6_9",
            # V5 merges former V7 and V8
            "V5_1": "V7_1", "V5_2": "V7_2", "V5_3": "V7_3",
            "V5_4": "V8_1", "V5_5": "V8_2", "V5_6": "V8_3", "V5_7": "V8_4", "V5_8": "V8_5",
            "V5_9": "V8_6", "V5_10": "V8_7", "V5_11": "V8_8", "V5_12": "V8_9", "V5_13": "V8_10"
        }
        
        # Reverse mapping: old name -> new name
        self.old_to_new_mapping = {v: k for k, v in self.new_to_old_mapping.items()}
        
        # New conflict group definitions
        self.conflict_groups = {
            # Major classes are not mutually exclusive; only minor variants within each class conflict
            "V1": ["V1_2", "V1_3", "V1_4", "V1_5", "V1_6"],  # V1_1 is default, not in conflict group
            "V2": ["V2_1", "V2_2", "V2_3", "V2_4", "V2_5"],
            "V3": ["V3_1", "V3_2", "V3_3", "V3_4", "V3_5", "V3_6", "V3_7", "V3_8", "V3_9", "V3_10", 
                   "V3_11", "V3_12", "V3_13", "V3_14", "V3_15", "V3_16", "V3_17", "V3_18", "V3_19", "V3_20"],
            "V4_1": ["V4_1_1", "V4_1_2", "V4_1_3", "V4_1_4", "V4_1_5", "V4_1_6", "V4_1_7"],
            "V4_2": ["V4_2_1", "V4_2_2", "V4_2_3"],
            "V4_3": ["V4_3_1", "V4_3_2", "V4_3_3"],
            "V4_4": ["V4_4_1", "V4_4_2", "V4_4_3"],
            "V5": ["V5_1", "V5_2", "V5_3", "V5_4", "V5_5", "V5_6", "V5_7", "V5_8", "V5_9", "V5_10", "V5_11", "V5_12", "V5_13"],
            
            # V4 special conflict: V4_1 conflicts with V4_2/V4_3/V4_4, but V4_2/V4_3/V4_4 do not conflict with each other
            "V4_DELIMITER_CONFLICT": ["V4_1_1", "V4_1_2", "V4_1_3", "V4_1_4", "V4_1_5", "V4_1_6", "V4_1_7",
                                     "V4_2_1", "V4_2_2", "V4_2_3", "V4_3_1", "V4_3_2", "V4_3_3", "V4_4_1", "V4_4_2", "V4_4_3"]
        }
        
        # New class configuration
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
                    logging.warning(f"Variant {variant} is in conflict_groups but not defined in class_config")

class VariationProbabilityManager:
    """Manages major and minor class variation probabilities."""
    def __init__(self, config: VariationConfig):
        self.config = config
        
        # Ensure random seed is initialized
        if not RandomSeedManager.is_initialized():
            RandomSeedManager.set_seed()
            logging.debug("VariationProbabilityManager: auto-initialized random seed")
        # Initialize 8 major class probabilities (new grouping structure)
        self.major_probs = {
            "V1": 1/8, "V2": 1/8, "V3": 1/8, "V4_1": 1/8, "V4_2": 1/8, "V4_3": 1/8, "V4_4": 1/8, "V5": 1/8
        }
        self.minor_probs = {}
        for class_name in config.class_config:
            variants = [v for idx, v in config.class_config[class_name].items() if idx != 0]
            # Use new variant names directly; no mapping needed
            class_variants = {}
            for v in variants:
                variant_name = v[0]  # Get variant name
                class_variants[variant_name] = 1/len(variants)
            self.minor_probs[class_name] = class_variants
        logging.debug("Initialized variation probabilities: major=%s, minor=%s", self.major_probs, self.minor_probs)

    def initialize_probabilities(self, scores: dict):
        """Initialize probabilities from initial experiment scores."""
        logging.info("Initializing variation probabilities")
        # Compute major class average scores
        major_scores = {class_name: 0 for class_name in self.major_probs}
        major_counts = {class_name: 0 for class_name in self.major_probs}
        for variant, score in scores.items():
            class_name = self.config.variant_to_class.get(variant, None)
            if class_name:
                major_scores[class_name] += score
                major_counts[class_name] += 1
                logging.debug("Variant %s (class %s) score: %.4f", variant, class_name, score)
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
        logging.info("Major class probabilities: %s", self.major_probs)
        # Set minor class probabilities
        for class_name in self.minor_probs:
            variants = self.minor_probs[class_name]
            num_variants = len(variants)
            for variant in variants:
                variants[variant] = round(1 / num_variants, 2)
            logging.info("Minor class probabilities (%s): %s", class_name, variants)

    def update_probabilities(self, parent_variants: list, new_variant: str, old_score: float, new_score: float, seed_pool_scores: list = None, learning_rate: float = 2):
        """Dynamically adjust probabilities based on variation effectiveness."""
        logging.debug("Updating probabilities: parent_variants=%s, new_variant=%s, old_score=%.4f, new_score=%.4f", parent_variants, new_variant, old_score, new_score)
        # Compute dynamic baseline score
        BASELINE_SCORE = min(seed_pool_scores) if seed_pool_scores else 0.5840
        logging.debug("Dynamic baseline score: %.4f", BASELINE_SCORE)
    
        # Compute change ratios
        parent_change = (new_score - old_score) / max(old_score, 1e-6)
        baseline_change = (new_score - BASELINE_SCORE) / max(BASELINE_SCORE, 1e-6) if new_score > old_score else 0
        if new_score > old_score:
            adjustment = 1 + learning_rate * baseline_change + learning_rate * parent_change
            adjustment = min(max(adjustment, 1.0), 2.0)
            logging.debug("Score improved, adjustment factor=%.2f", adjustment)
            variants_to_update = parent_variants + [new_variant] if new_score > BASELINE_SCORE else [new_variant]
        else:
            adjustment = 1 - learning_rate * abs(parent_change)
            adjustment = min(max(adjustment, 0.5), 1.0)
            logging.debug("Score decreased, adjustment factor=%.2f", adjustment)
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
                # Use new variant names directly
                if variant in variants:
                    total_prob = sum(variants.values())
                    variants[variant] *= adjustment
                    new_total = sum(variants.values())
                    for k in variants:
                        variants[k] = round(variants[k] * total_prob / new_total, 2)
                else:
                    logging.warning(f"Variant {variant} not found in minor_probs[{class_name}]")
                updated_classes.add(class_name)

    def select_variation(self, seed_classes: list = None) -> str:
        """Select a variation based on probabilities."""
        classes = list(self.major_probs.keys())
        probs = [self.major_probs[c] for c in classes]
        if seed_classes:
            classes = [c for c in classes if c in seed_classes]
            probs = [self.major_probs[c] for c in classes]
            if not classes:
                logging.warning("No available seed classes: %s", seed_classes)
                return None
        class_name = np.random.choice(classes, p=np.array(probs) / sum(probs))
        variants = list(self.minor_probs[class_name].keys())
        probs = [self.minor_probs[class_name][v] for v in variants]
        selected_variant = np.random.choice(variants, p=np.array(probs) / sum(probs))
        
        logging.debug("Selected variation: class=%s, variant=%s", class_name, selected_variant)
        return selected_variant
    
    def set_random_seed(self, seed: int = None):
        """
        Set random seed for probability manager.
        Args:
            seed: Seed value; if None, generated from timestamp
        Returns:
            Actual seed value used
        """
        actual_seed = RandomSeedManager.set_seed(seed)
        logging.info(f"🎲 VariationProbabilityManager: random seed set to {actual_seed}")
        return actual_seed
    
    def get_current_seed(self):
        """Get current random seed."""
        return RandomSeedManager.get_seed()

def check_v4_conflicts(variants, config):
    """Check V4 special conflicts: V4_1 is mutually exclusive with V4_2/V4_3/V4_4."""
    v4_1_variants = [v for v in variants if v in config.conflict_groups.get("V4_1", [])]
    v4_234_variants = []
    for v in variants:
        if (v in config.conflict_groups.get("V4_2", []) or 
            v in config.conflict_groups.get("V4_3", []) or 
            v in config.conflict_groups.get("V4_4", [])):
            v4_234_variants.append(v)
    
    # Conflict if V4_1 and V4_2/V4_3/V4_4 coexist
    return len(v4_1_variants) > 0 and len(v4_234_variants) > 0

def generate_random_template(max_variations, config, prob_manager: VariationProbabilityManager, seed_classes=None, parent_variants=None):
    # Ensure random seed is initialized
    if not RandomSeedManager.is_initialized():
        RandomSeedManager.set_seed()
        logging.debug("generate_random_template: auto-initialized random seed")
    
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
    # In new structure, major classes do not conflict; only check intra-class conflicts
    conflict_classes = [c for c, count in class_counts.items() if count > 1]
    conflict_groups = []
    for group, group_variants in config.conflict_groups.items():
        group_count = sum(1 for variant in all_variants if variant in group_variants)
        if group_count > 1:
            conflict_groups.append((group, [v for v in all_variants if v in group_variants]))
    
    # Check V4 special conflicts
    v4_conflict = check_v4_conflicts(all_variants, config)
    
    if conflict_classes or conflict_groups or v4_conflict:
        return {class_name: [] for class_name in config.class_config}, []

    def variant_to_class_index(variant, class_name):
        """Convert variant name to corresponding class index."""
        # Handle new naming rules
        parts = variant.split("_")
        if len(parts) == 2:
            # V1_1, V2_1, etc.
            variant_idx = int(parts[1])
            return variant_idx
        elif len(parts) == 3:
            # V4_1_1, V4_2_1, etc.
            variant_idx = int(parts[2])
            return variant_idx
        else:
            # Other cases: try to extract index from last part
            try:
                variant_idx = int(parts[-1])
                return variant_idx
            except ValueError:
                logging.warning(f"Unable to parse variant index: {variant}")
                return 1  # Default to 1

    template = {class_name: [] for class_name in config.class_config}
    for variant in selected_variants:
        class_name = config.variant_to_class[variant]
        variant_idx = variant_to_class_index(variant, class_name)
        template[class_name].append(variant_idx)

    return template, selected_variants

class RandomMutationGenerator:
    """Fully random variation combination generator."""
    def __init__(self, config: 'VariationConfig'):
        self.config = config
        self.used_combinations = set()  # Track used variation combinations
        
        # Ensure random seed is initialized
        if not RandomSeedManager.is_initialized():
            RandomSeedManager.set_seed()
            logging.debug("RandomMutationGenerator: auto-initialized random seed")
        
    def generate_random_combination(self, max_variations: int) -> tuple:
        """
        Generate a fully random variation combination.
        Improved logic:
        1. Use specified major class selection probabilities:
           V1: 1/8, V2: 1/8, V3: 1/8, V4: 1/8, V5: 1/16,
           V6_BOS: 1/16, V6_BOT: 1/16, V6_ROLE: 1/16, V7: 1/8, V8: 1/8
        2. All minor variants within a class have equal probability
        
        Args:
            max_variations: Maximum number of stacked variations
        Returns:
            (template_dict, variants_list) or (None, None) if no valid combination can be generated
        """
        max_attempts = 1000  # Max attempts to avoid infinite loop
        
        # Major class selection probabilities (new grouping structure)
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
            # Get all available major classes
            available_classes = list(self.config.class_config.keys())
            
            if not available_classes:
                logging.warning("🎲 Random mutation mode: no available major classes")
                return None, None
            
            # Randomly select number of variations (1 to max_variations)
            num_variations = random.randint(1, max_variations)
            
            # New selection logic: select major class first, then minor variant
            selected_variants = []
            selected_classes = set()
            
            # Track selected variants per conflict group
            conflict_group_selections = {group: [] for group in self.config.conflict_groups}
            
            for _ in range(num_variations):
                # Step 1: Get selectable major classes (exclude already selected and conflicting ones)
                available_classes_for_selection = []
                available_probabilities = []
                
                for class_name in available_classes:
                    if class_name in selected_classes:
                        continue
                    
                    # Check if class has available variants (excluding conflicts)
                    class_has_available_variants = False
                    for idx, variant_list in self.config.class_config[class_name].items():
                        if idx == 0 or not variant_list:  # Skip no-variation option
                            continue
                        
                        variant = variant_list[0]  # Get variant name
                        
                        # Check mutual exclusion
                        can_select = True
                        for group, group_variants in self.config.conflict_groups.items():
                            if variant in group_variants:
                                # Check if conflict group already has a selection
                                if conflict_group_selections[group]:
                                    can_select = False
                                    break
                        
                        # Check V4 special conflicts
                        if can_select:
                            # Build temporary variant list for V4 conflict check
                            temp_variants = selected_variants + [variant]
                            if check_v4_conflicts(temp_variants, self.config):
                                can_select = False
                        
                        if can_select:
                            class_has_available_variants = True
                            break
                    
                    if class_has_available_variants:
                        available_classes_for_selection.append(class_name)
                        # Get class probability, or use default
                        prob = class_probabilities.get(class_name, 1/16)
                        available_probabilities.append(prob)
                
                if not available_classes_for_selection:
                    break
                
                # Step 2: Select a major class by specified probability
                # Normalize probabilities
                total_prob = sum(available_probabilities)
                normalized_probs = [p / total_prob for p in available_probabilities]
                
                # Weighted random selection of major class
                selected_class = random.choices(available_classes_for_selection, weights=normalized_probs)[0]
                logging.debug(f"🎲 Selected major class by probability: {selected_class} (prob: {class_probabilities.get(selected_class, 1/16):.4f})")
                
                # Step 3: Within selected class, equally likely minor variant selection
                available_variants_in_class = []
                for idx, variant_list in self.config.class_config[selected_class].items():
                    if idx == 0 or not variant_list:  # Skip no-variation option
                        continue
                    
                    variant = variant_list[0]  # Get variant name
                    
                    # Check mutual exclusion
                    can_select = True
                    for group, group_variants in self.config.conflict_groups.items():
                        if variant in group_variants:
                            # Check if conflict group already has a selection
                            if conflict_group_selections[group]:
                                can_select = False
                                break
                    
                    # Check V4 special conflicts
                    if can_select:
                        # Build temporary variant list for V4 conflict check
                        temp_variants = selected_variants + [variant]
                        if check_v4_conflicts(temp_variants, self.config):
                            can_select = False
                    
                    if can_select:
                        available_variants_in_class.append(variant)
                
                if not available_variants_in_class:
                    continue
                
                # Equally likely random minor variant within class
                selected_variant = random.choice(available_variants_in_class)
                logging.debug(f"🎲 Randomly selected minor variant in class {selected_class}: {selected_variant}")
                
                selected_variants.append(selected_variant)
                selected_classes.add(selected_class)
                
                # Update conflict group selections
                for group, group_variants in self.config.conflict_groups.items():
                    if selected_variant in group_variants:
                        conflict_group_selections[group].append(selected_variant)
            
            if not selected_variants:
                continue
            
            # Generate combination signature for deduplication
            combination_signature = tuple(sorted(selected_variants))
            
            # Check for duplicates
            if combination_signature in self.used_combinations:
                continue
            
            # Record used combination
            self.used_combinations.add(combination_signature)
            
            # Convert to template format
            template = self._variants_to_template(selected_variants)
            
            logging.info(f"🎲 Random mutation mode: generated new combination (attempt {attempt + 1}): {selected_variants}")
            logging.debug(f"🎲 Selection process: major class by specified probability -> equal minor class probability")
            logging.debug(f"🎲 Major class probabilities: V1/V2/V3/V4/V7=1/8, V5/V6_BOS/V6_BOT/V6_ROLE/V8_PROMPT/V8_SEP=1/16")
            return template, selected_variants
        
        logging.warning(f"🎲 Random mutation mode: failed to generate new random combination after {max_attempts} attempts")
        return None, None
    
    def _variants_to_template(self, variants: list) -> dict:
        """Convert variant list to template dictionary format."""
        def variant_to_class_index(variant, class_name):
            """Convert variant name to corresponding class index."""
            # Handle new naming rules
            parts = variant.split("_")
            if len(parts) == 2:
                # V1_1, V2_1, etc.
                variant_idx = int(parts[1])
                return variant_idx
            elif len(parts) == 3:
                # V4_1_1, V4_2_1, etc.
                variant_idx = int(parts[2])
                return variant_idx
            else:
                # Other cases: try to extract index from last part
                try:
                    variant_idx = int(parts[-1])
                    return variant_idx
                except ValueError:
                    logging.warning(f"Unable to parse variant index: {variant}")
                    return 1  # Default to 1
        
        template = {class_name: [] for class_name in self.config.class_config}
        for variant in variants:
            class_name = self.config.variant_to_class[variant]
            variant_idx = variant_to_class_index(variant, class_name)
            template[class_name].append(variant_idx)
        
        return template
    
    def reset_used_combinations(self):
        """Reset used combination records."""
        self.used_combinations.clear()
        logging.info("🎲 Random mutation mode: reset variation combination records")
    
    def get_used_combinations_count(self) -> int:
        """Get count of used combinations."""
        return len(self.used_combinations)
    
    def set_random_seed(self, seed: int = None):
        """
        Set random seed for current generator.
        Args:
            seed: Seed value; if None, generated from timestamp
        Returns:
            Actual seed value used
        """
        actual_seed = RandomSeedManager.set_seed(seed)
        logging.info(f"🎲 RandomMutationGenerator: random seed set to {actual_seed}")
        return actual_seed
    
    def get_current_seed(self):
        """Get current random seed."""
        return RandomSeedManager.get_seed()
    
    def reset_with_new_seed(self):
        """Reset generator with new random seed and clear used combination records."""
        new_seed = RandomSeedManager.reset_seed()
        self.used_combinations.clear()
        logging.info(f"🎲 RandomMutationGenerator: reset with new seed {new_seed}, cleared combination records")
        return new_seed


# Convenience functions for external callers
def set_variation_random_seed(seed: int = None):
    """
    Set random seed for variation module.
    Args:
        seed: Seed value; if None, generated from timestamp
    Returns:
        Actual seed value used
    """
    return RandomSeedManager.set_seed(seed)

def get_variation_random_seed():
    """Get current random seed for variation module."""
    return RandomSeedManager.get_seed()

def reset_variation_random_seed():
    """Reset random seed for variation module (using new timestamp)."""
    return RandomSeedManager.reset_seed()

def is_variation_seed_initialized():
    """Check whether variation module random seed has been initialized."""
    return RandomSeedManager.is_initialized()
