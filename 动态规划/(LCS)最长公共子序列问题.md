<font face= "楷体" size = 3>

<center><font face="楷体" size=6, color='red'> 最长公共子序列(LCS) </font> </center>


### 1143. 最长公共子序列

```c++
class Solution {
public:
    int longestCommonSubsequence(string s1, string s2) {
        int n = s1.size(), m = s2.size();
        vector<vector<int>>f(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; i++) 
            for (int j = 1; j <= m; j++) 
                if (s1[i - 1] == s2[j - 1]) f[i][j] = f[i - 1][j - 1] + 1;
                else f[i][j] = max(f[i - 1][j], f[i][j - 1]);
        return f[n][m];
    }
};
```
---

### 1035. 不相交的线
题意：两个数组中的数字写成两行，相等的数字可以上下连线，每个数字只能连一次，问最多可以连多少条不想交的线。

**代码同上**

---

### 583. 两个字符串的删除操作
给定两个单词 `word1` 和` word2 `，返回使得 `word1`和 `word2` 相同所需的最小步数。
`word1 = "sea", word2 = "eat", ans = 2`


#### 思路1：DP
**状态表示**：`dp[i][j]`表示字符串`1`的前`i`个字符与字符串`2`的前`j`个字符相同需要删除的最小步数
**状态转移**: 
(1):`word1[i - 1] == word2[j - 1]` (索引从0开始):`dp[i][j] = dp[i - 1][j - 1]` (不用删除)
(2):`word1[i - 1] != word2[j - 1]`: 在`dp[i - 1][j]`的基础上删掉`word2[j - 1]`或者在`dp[i][j - 1]`的基础上删掉`word1[i - 1]`
        
```c++
class Solution {
public:
    int minDistance(string w1, string w2) {
        int n = w1.size(), m = w2.size();
        vector<vector<int>>f(n + 1, vector<int>(m + 1));
        for (int i = 0; i <= n; i++) f[i][0] = i;
        for (int j = 0; j <= m; j++) f[0][j] = j;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (w1[i - 1] == w2[j - 1]) f[i][j] = f[i - 1][j - 1];
                else f[i][j] = min(f[i - 1][j], f[i][j - 1]) + 1;
            }
        }
        return f[n][m];
    }
};
```


#### 思路2：先求LCS
```c++
class Solution {
public:
    int minDistance(string w1, string w2) {
        int n = w1.size(), m = w2.size();
        vector<vector<int>>f(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; i++) 
            for (int j = 1; j <= m; j++)
                if (w1[i - 1] == w2[j - 1]) f[i][j] = f[i - 1][j - 1] + 1;
                else f[i][j] = max(f[i - 1][j], f[i][j - 1]);
        int lcs = f[n][m];
        return n + m - 2 * lcs;
    }
};
```
---