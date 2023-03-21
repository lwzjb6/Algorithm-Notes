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