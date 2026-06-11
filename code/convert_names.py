from pybaseball import chadwick_register

def convert_names(ids,register=None):
    if register is None:
        register = chadwick_register()
    if type(ids)==int or type(ids)==float:
        names = f"{register[register.key_mlbam==ids].name_first.values[0]} {register[register.key_mlbam==ids].name_last.values[0]}"
    else:
        skipped = []
        names = []
        for i in ids:
            try:
                names.append(f"{register[register.key_mlbam==i].name_first.values[0]} {register[register.key_mlbam==i].name_last.values[0]}")
            except IndexError:
                names.append('None')
                skipped.append(i)
        print(f"{len(skipped)}/{len(ids)} skipped")
    return names

def main():
    data = chadwick_register()
    ids_1 = 605152
    ids_2 = [605152,110625,118336]
    print(convert_names(ids_1,data))
    print(convert_names(ids_2,data))

if __name__=="__main__":
    main()