<font face= "楷体" size = 3>
---
<center><font face="楷体" size=6, color='red'> 区间DP </font> </center>


### 5. 最长回文子串
返回`s`的最长回文字串

#### 固定中间点

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        string ans = "";
        int n = s.size();
        for (int i = 0; i < n; i++) {
            int l = i, r = i;
            while(l >= 0 && r < n && s[l] == s[r]) {
                if(r - l + 1 > ans.size()) ans = s.substr(l, r - l + 1);
                l--, r++;
            }
        }
        for (int i = 0; i < n; i++) {
            int l = i, r = i + 1;
            while(l >= 0 && r < n && s[l] == s[r]) {
                if(r - l + 1 > ans.size()) ans = s.substr(l, r - l + 1);
                l--, r++;
            }
        }
        return ans;
    }
};
```
#### 区间DP
预处理`f[i][j]`是否是回文的
然后找到最大的，时间复杂度$O(n^2)$

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
        vector<vector<int>>f(n, vector<int>(n, 1));
        string ans = string(1, s[0]);
        for(int i = n - 1; i >= 0; i--) {
            for(int j = i + 1; j < n; j++) {
                f[i][j] = f[i + 1][j - 1] && (s[i] == s[j]);
                if(f[i][j] && j - i + 1 > ans.size()) ans = s.substr(i, j - i + 1); 
            }
        }
        return ans;
    }
};
```
---

### AcWing 282. 石子合并
将`N`堆石子合并为`1`堆，每次只能合并相邻的两堆,合并的代价为这两堆石子的质量之和, 问合并的最小代价是多少。

状态表示：`f[i][j]` 表示合并`i~j`区间的石子所付出的代价
状态转移：`f[i][j] = min{f[i][k], f[k + 1][j]} + s[j] - s[i - 1]`;
状态初始化：`f[i][i] = 0` , 其他为无穷大

```c++
#include<bits/stdc++.h>
using namespace std;
int main(){
    int n;
    cin >> n;
    vector<int>a(n);
    for(int i = 0; i < n; i++) cin >> a[i];
    
    vector<int>s(n + 1, 0); // 前缀和
    for(int i = 1; i <= n; i++) s[i] = s[i - 1] + a[i - 1];
    
    int f[n + 1][n + 1]; 
    memset(f, 0x3f, sizeof(f));
    
    for(int len = 1; len <= n; len++) { // 第一维通常是枚举区间长度
        for(int i = 1; i + len - 1 <= n; i++) { // 第二维枚举区间起点, 同时保证终点j不会越界
            int j = i + len - 1;
            if(i == j) f[i][j] = 0;
            for (int k = i; k + 1 <= j; k++) {// 第三维枚举分割点
                f[i][j] = min(f[i][j], f[i][k] + f[k + 1][j] + s[j] - s[i - 1]);
            }
        }
    }
    cout << f[1][n] << endl;
    return 0;
}
```
---

### 1000. 合并石头的最低成本
和上题基本一样，区别在于每次是合并相邻的K堆

```c++
class Solution {
public:
    int mergeStones(vector<int>& stones, int k) {
        int n = stones.size();
        int f[n][n];
        memset(f, 0x3f, sizeof(f));
        if((n - 1) % (k - 1)) return -1;

        vector<int>s(n + 1, 0); // 前缀和
        for(int i = 1; i <= n; i++) s[i] = s[i - 1] + stones[i - 1];

        // 初始化
        for(int i = 0; i < n; i++) f[i][i] = 0;

        for(int len = 2; len <= n; len++) {
            for(int i = 0; i + len - 1 < n; i++) {
                int j = i + len - 1;
                for(int m = i; m < j; m += k - 1) { // 枚举分割点
                    f[i][j] = min(f[i][j], f[i][m] + f[m + 1][j]);
                }
                if((len - 1) % (k - 1) == 0) f[i][j] += s[j + 1] - s[i]; // 可以合并为一堆
            }
        }
        return f[0][n - 1];
    }
};
```


### 312. 戳气球
`n`个气球，每隔都有一个价值，打破一个气球的收益为`nums[i - 1] * nums[i] * nums[i + 1]` 超出边界的气球价值为1，问最大的收益。
```c++
nums = [3,1,5,8]
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
```


**小技巧:** 将原始左右两边各加一个价值为1的气球

状态表示：`f[i][j]:`打破`i~j`（不包括边界）之间的气球的最大收益
结果： `f[1][n]`因为添加了左右两边各一个
状态转移：`f[i][j] = max(nums[i] * nums[k] * nums[j] + f[i][k] + f[k][j]) for k in range(i + 1, j)` 假设打破`k`处的气球。

```c++
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        // add 2 dummy node
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        
        int f[n + 2][n + 2];
        memset(f, 0, sizeof f);

        for(int len = 2; len <= n + 2; len++) { // 枚举长度
            for(int i = 0; i + len - 1 < n + 2; i++) { // 枚举起点
                int j = i + len - 1;
                for(int k = i + 1; k < j; k++) { // 枚举打破的点
                    f[i][j] = max(f[i][j], nums[i] * nums[k] * nums[j] + f[i][k] + f[k][j]);
                }
            }
        }
        return f[0][n + 1];
    }
}; 
```
---


### 1039. 多边形三角剖分的最低得分
将一个凸多边形划分为`n - 2`个三角形，问划分的最低得分为多少，经典的三角剖分问题。

#### 记忆化搜索
```c++
class Solution {
public:
    int minScoreTriangulation(vector<int>& values) {
        int n = values.size();
        int f[n][n];
        memset(f, -1, sizeof(f));
        // 从顶点i顺时针走到顶点j多边形的最小得分
        function<int(int, int)>DFS = [&](int i, int j){
            if(i + 1 == j) return 0; // 相邻的情况
            if(f[i][j] != -1) return f[i][j];
            int res = INT_MAX;
            for(int k = i + 1; k < j; k++) { // 划分子问题
                res = min(res, DFS(i, k) + DFS(k , j) + values[i] * values[j] * values[k]);
            }
            f[i][j] = res;
            return res;
        };
        return DFS(0, n - 1); // 答案
    }
};
```
**时间复杂度 = 状态个数 * 每个状态的计算时间 = $O(n^2) O(n) = O(n^3)$**

#### 区间DP

状态表示：`f[i][j]: 将区间i, j之间的多边形进行剖分的最小得分`
状态转移：`f[i][j] = min{f[i][k], f[k][j], a[i] * a[j] * a[k]} for k in range(i + 1, j)` [枚举所有的分割点]
结果： `f[0][n - 1]`

注意点：

```c++
class Solution {
public:
    int minScoreTriangulation(vector<int>& values) {
        int n = values.size();
        int f[n][n];
        memset(f, 0, sizeof f);

        for(int len = 3; len <= n; len++) { // 枚举长度
            for(int i = 0; i + len - 1 < n; i++){ // 枚举起点
                int j = i + len - 1;
                f[i][j] = INT_MAX;
                for(int k = i + 1; k < j; k++) { // 枚举分割点
                    f[i][j] = min(f[i][k] + f[k][j] + values[i] * values[j] * values[k], f[i][j]);
                }
            }
        }
        return f[0][n - 1];
    }
};
```
---

### 375. 猜数字大小 II
我从 `1` 到 `n` 之间选择一个数字。你来猜我选了哪个数字。
如果猜`x`,但是猜错了，需要支付`x`，问在保证可以猜对的情况下最少支付多少钱。

#### 区间DP
```c++
class Solution {
public:
    int getMoneyAmount(int n) {
        vector<vector<int>>dp(n + 1, vector<int>(n + 1, 0));
        // dp[i][j]：猜区间i~j之间的数字，最少需要支付多少。  dp[i][i] = 0
        for(int len = 2; len <= n; len++) {
            for(int i = 1; i + len - 1 <= n; i++) {
                int j = i + len - 1;
                dp[i][j] = min(i + dp[i + 1][j], j + dp[i][j - 1]); // 边界
                for(int k = i + 1; k < j; k++) { // 以k作为分割点
                    dp[i][j] = min(dp[i][j], k + max(dp[i][k - 1], dp[k + 1][j]));
                }
            }
        }
        return dp[1][n];
    }
};
```
---

### 1312. 让字符串成为回文串的最少插入次数
返回让 `s` 成为回文串的 最少操作次数
`s = "leetcode", ans = 5`

#### LCS
`t = reverse(s)`
求`s, t`的`LCS`
`ans = n - LCS`

#### 区间DP
状态表示：`f[i][j]:` 让字符串`s`区间`i,j`之间变为回文串的最少操作次数

状态转移：
(1) `s[i] == s[j]`:
`f[i][j] = min(f[i + 1][j - 1], f[i][j - 1] + 1, f[i + 1][j] + 1)`
(2) `s[i] != s[j] `:
`f[i][j] = min(f[i][j - 1] + 1, f[i + 1][j] + 1)`

```c++
class Solution {
public:
    int minInsertions(string s) {
        int n = s.size();
        int f[n][n];
        memset(f, 0, sizeof(f));
        for(int len = 2; len <= n; len++) {
            for(int i = 0; i + len - 1 < n; i++) {
                int j = i + len - 1;
                f[i][j] = min(f[i + 1][j], f[i][j - 1]) + 1;
                if(s[i] == s[j]) f[i][j] = min(f[i][j], f[i + 1][j - 1]);
            }
        }
        return f[0][n - 1];
    }
};
```
---

