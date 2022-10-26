import random
import math

class Choice:
    def __init__(self, choices):
        self.choices = choices
    
    def __call__(self):
        return random.choice(self.choices)
    
class Fixed:
    def __init__(self, choices):
        self.choice = random.choice(choices)
    
    def __call__(self):
        return self.choice

class Pronunciation:
    def __init__(self, is_alphanumeric):
        self.mappings = {
            # Letters
            "a": Choice(["ae"]),
            "b": Choice(["bee"]),
            "c": Choice(["see"]),
            "d": Choice(["dee"]),
            "e": Choice(["ee"]),
            "f": Choice(["ef"]),
            "g": Choice(["jee"]),
            "h": Fixed(["h", "haych", "aeych"]),
            "i": Choice(["eye", "i"]),
            "j": Choice(["jay", "j"]),
            "k": Choice(["kay", "k"]),
            "l": Choice(["el", "l"]),
            "m": Fixed(["am", "em", "m"]),
            "n": Fixed(["aen", "en", "n"]),
            "o": Choice(["o", "oo", "oh"]),
            "p": Choice(["pee", "p"]),
            "q": Choice(["cuu", "q"]),
            "r": Choice(["aar", "arr", "r"]),
            "s": Choice(["s", "es"]),
            "t": Choice(["tee", "ti"]),
            "u": Choice(["you", "uu"]),
            "v": Choice(["wee", "v"]),
            "w": Choice(["double you", "double u"]),
            "x": Choice(["eks", "x"]),
            "y": Fixed(["why", "wai", "y"]),
            "z": Fixed(["zee", "z", "zed"]),
            #Numbers
            "0": Fixed(["zero", "o"]) if not is_alphanumeric else Fixed(["zero"]),
            "1": Choice(["one"]),
            "2": Fixed(["two", "too"]),
            "3": Choice(["three"]),
            "4": Fixed(["four", "fo"]),
            "5": Fixed(["five", "faaev"]),
            "6": Choice(["six"]),
            "7": Choice(["seven"]),
            "8": Choice(["eight"]),
            "9": Choice(["nine"]),   
        }

    # def get_inverse_mapping(self):
    #     d = {}
    #     for k, v in self.mappings.items():
    #         for vi in v.choices:
    #             d[vi] = k
    #     return d

    def denormalize(self, norm_string):
        pass

    # def normalize(self, denorm_string):
    #     pass

class GenericPronunciation(Pronunciation):
    '''
    Sequence"        "aabccc001"
    Pronunciation:   "ae ae bee see see zero zero one"
    '''
    def __init__(self, is_alphanumeric):
        super().__init__(is_alphanumeric)
        # self.inverse_mappings = self.get_inverse_mapping()    

    # def normalize(self, denorm_string):
    #     return "".join([self.inverse_mappings[c] for c in denorm_string.split(" ")])

    def denormalize(self, norm_string):
        return " ".join([self.mappings[c]() for c in norm_string])
        
class GroupedGenericPronunciation(GenericPronunciation):
    '''
    Sequence"        "aabccc001"
    Pronunciation:   "double ae bee triple see double zero one"
    '''
    def __init__(self, is_alphanumeric, group_alpha):
        super().__init__(is_alphanumeric)
        if group_alpha:
            # we don't say "triple double you" for pronuncing "WWW", instead we say "double you double you double you"
            # hence don't group "w"
            self.to_group = {k: k != "w" for k in self.mappings.keys()}
        else:
            self.to_group = {k: not k.isalpha() for k in self.mappings.keys()}

    def grouped_pronunciation(self, c, f):
        if f == 0:
            return ""
        elif f == 1:
            return self.mappings[c]()
        elif f == 2:
            return f"double {self.mappings[c]()}"
        elif f == 3:
            return f"triple {self.mappings[c]()}"
        elif f == 4:
            return random.choice([
                f"double {self.mappings[c]()} double {self.mappings[c]()}",
                f"four times {self.mappings[c]()}",
            ])
        elif f == 5:
            return random.choice([
                f"triple {self.mappings[c]()} double {self.mappings[c]()}",
                f"double {self.mappings[c]()} triple {self.mappings[c]()}",
                f"five times {self.mappings[c]()}"
            ])
        elif f == 6:
            return random.choice([
                f"triple {self.mappings[c]()} triple {self.mappings[c]()}",
                f"six times {self.mappings[c]()}",
            ])
        else:
            return " ".join([self.grouped_pronunciation(c, 6) for _ in range(math.floor(f/6))] + [self.grouped_pronunciation(c, f%6)])        

    def denormalize(self, norm_string):
        frequency_table = []
        chars = []

        # count repeats of characters
        last_c = norm_string[0]
        last_count = 1
        for c in norm_string[1:]:
            if c == last_c:
                last_count += 1
            else:
                frequency_table.append(last_count)
                chars.append(last_c)
                
                last_c = c
                last_count = 1

        frequency_table.append(last_count)
        chars.append(c)
        # print(chars, frequency_table)

        words = []
        for c, f in zip(chars, frequency_table):
            if self.to_group[c]:
                words.append(self.grouped_pronunciation(c, f))
            else:
                words.extend([self.mappings[c]() for _ in range(f)])
        return " ".join(words)

def build_pronounciations(classname, is_alphanumeric, num_objs, *args, **kwargs):
    all_objs = []
    for _ in range(num_objs):
        o = eval(classname)(is_alphanumeric, *args, **kwargs)
        all_objs.append(o)
    return all_objs

def get_pron_objs(dataset, num_examples):
    if dataset == "numeric":
        objs1 = build_pronounciations("GenericPronunciation", False, num_examples // 2)
        objs2 = build_pronounciations("GroupedGenericPronunciation", False, num_examples // 2, group_alpha=False)
        total = objs1 + objs2
    elif dataset == "alpha":
        objs1 = build_pronounciations("GenericPronunciation", False, num_examples // 2)
        objs2 = build_pronounciations("GroupedGenericPronunciation", False, num_examples // 2, group_alpha=True)
        total = objs1 + objs2
    elif dataset == "alphanumeric":
        objs1 = build_pronounciations("GenericPronunciation", True, num_examples // 2)
        objs2 = build_pronounciations("GroupedGenericPronunciation", True, num_examples // 4, group_alpha=False)
        objs3 = build_pronounciations("GroupedGenericPronunciation", True, num_examples - num_examples // 4 - num_examples // 2, group_alpha=True)
        total = objs1 + objs2 + objs3
    assert(len(total) == num_examples)
    return total
        
if __name__ == "__main__":
    sequence = "600444"

    s1 = GenericPronunciation(True).denormalize(sequence)
    s2 = GroupedGenericPronunciation(True, group_alpha=False).denormalize(sequence)
    s3 = GroupedGenericPronunciation(True, group_alpha=True).denormalize(sequence)
    # print("normalized", sequence)
    # print("generic", s1)
    # print("generic-grouped", s2)
    # print("generic-grouped alpha", s3)

    os = build_pronounciations("GroupedGenericPronunciation", False, 5, group_alpha=False)
    print([o.denormalize(sequence) for o in os])
    # os = get_pron_objs("alphanumeric", 1000)
    print([o.denormalize("00688") for o in os])
