
_Str = 'goooog'


def solve(s):
    _len = len(s)
    _ret = 1

    dp = [
        [0 for i in range(_len)]
        for j in range(_len)
    ]

    for i in range(_len): dp[i][i] = 1

    for i in range(1, _len):
        for j in range(_len):
            if j + i >= _len: break
            if s[j] == s[i + j]:
                dp[j][i+j]=dp[j+1][j+i-1] + 2
                _ret = max(_ret, dp[j][i+j])

    return _ret


if __name__ == '__main__':
    print(solve(_Str))

