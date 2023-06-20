<font face= "楷体" size = 3>
---
<center><font face="楷体" size=6, color='red'> 博弈DP </font> </center>

### 464. 我能赢吗
一堆数字卡片`1~n`,博弈双方每次从中选一个[每个数字只能选一次]，当某一方进行操作后，双方的已拿数字总和大于等于`t`时，当前方获胜。问是否先手获胜。

状态表示：`f[state]:` 在状态`state`时，当前方是否必胜。
核心判断： (1) 转移过程中，如果发现当前回合的决策，能够直接使得累积和超过`t`，说明当前回合玩家获胜；(2) 或者如果当前决策能够导致下一回合的玩家失败的话，当前回合玩家也获胜，否则当前玩家失败。

```c++
class Solution {
public:
    bool canIWin(int maxChoosableInteger, int desiredTotal) {
        int n = maxChoosableInteger, m = 1 << n;
        vector<int>f(m, -1); // f[i]: 状态i下当前下棋方是否必胜。
        // DFS(state, sum)：统计在当前状态为state，和为sum的情况下，当前方能否获胜的情况， 1表示必胜，0表示必输
        // 核心： (1) 转移过程中，如果发现当前回合的决策，能够直接使得累积和超过 desiredTotal，说明当前回合玩家获胜；
        //        (2) 或者如果当前决策能够导致下一回合的玩家失败的话，当前回合玩家也获胜，否则当前玩家失败。
        function<bool(int, int)>DFS = [&](int state, int sum){
            if(f[state] != -1) return f[state];
            for(int j = 0; j < n; j++) { // 枚举决策
                if((state >> j) & 1) continue; // 已经选过了
                if(sum + j + 1 >= desiredTotal) return f[state] = 1; // 找到一种胜利的策略
                if(DFS(state | (1 << j), sum + j + 1) == 0) return f[state] = 1; // 之后对方无法胜利
            }
            return f[state] = 0;
        };
        int mxsum = (1 + n) * n / 2;
        if(mxsum < desiredTotal) return 0;
        return DFS(0, 0);
    }
};
```
---
