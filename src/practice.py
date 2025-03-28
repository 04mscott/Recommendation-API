from collections import defaultdict

def max_var(s: str):
    # Each index in vars = max varience up to i in s
    vars = [0] * len(s)
    # get s[:i + 1]
    # get num occurances of each char
    # set vars[s] = max difference
    max_occur = 0
    max_char = ''
    for i in range(len(s)):
        sub_str = s[:i+1]
        nums = defaultdict(int)

        # Count num occurances of chars
        for c in sub_str:
            nums[c] += 1

            if nums[c] > max_occur:
                max_occur = nums[c]

                if c != max_char:
                    if max_char != '':
                        nums[max_char] = 1
                    max_char = c

        if len(nums) > 1:
            vars[i] = max(nums.values()) - min(nums.values())


    return max(vars)