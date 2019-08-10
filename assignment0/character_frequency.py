class character_frequency:
    def __init__(self):
        self.string = ''
    
    def character_frequency(self, string):
        character_dict = {}
        character = [x.strip() for x in string]
        for ch in character:
            if ch in character_dict:
                character_dict[ch] += 1
            else:
                character_dict[ch] = 1
        print(character_dict)
        return character_dict
