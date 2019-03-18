def update_array(array):
    result = []
    steps = array[0]
    for i in range(1, len(array)):
        if i - 1 < steps:
            result.append(array[i] - 1)
        else:
            result.append(array[i])
    result.sort(reverse=True)
    return result

def havel_hakimi(input):
    input.sort(reverse=True)
    if input[0] >= len(input):
        return False

    d1 = input[0]
    while d1 > 0:
        input = update_array(input)

        # if input is None:
        #     return True
        d1 = input[0]
        if min(input) < 0:
            return False
        elif d1 == 0:
            return True
    return False

if __name__ == '__main__':
    print(havel_hakimi([5,5,4,3,2,2,2,1]))
    print(havel_hakimi([5,5,4,4,2,2,1,1]))
    print(havel_hakimi([5,5,5,3,2,2,2,1,1]))
    print(havel_hakimi([5,5,5,4,2,1,1,1]))
