<font face="楷体" size = 3>

<center><font face="楷体" size=6, color='red'> DFS </font> </center>

---
### 77. 组合
给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的**组合**。
`eg: n = 4, k = 2`
`[[2,4],[3,4],[2,3],[1,2],[1,3],[1,4],]`

```c++
class Solution {
public:
    vector<vector<int>>ans;
    vector<int>res;
    // 当前考虑的是第u个数要不要选
    void dfs(int n, int k, int u) {
        // 剪枝
        if (res.size() + n - u + 1 < k) return;
        if (res.size() == k) {
            ans.push_back(res);
            return;
        }
        if (u == n + 1) return;
        // 不选数字u
        dfs(n, k, u + 1);
        // 选数字u
        res.push_back(u);
        dfs(n, k, u + 1);
        res.pop_back();
    }
    vector<vector<int>> combine(int n, int k) {
        dfs(n, k, 1);
        return ans;
    }
};
```
时间复杂度：$O(C^k_n * k)$
因为有剪枝的存在，每次向下搜索都能找到一个答案，而每次找答案案是k步。

---
### 216. 组合总和 III
求在数字`1-9`中选`k`个数组成和为`n`的所有方案
`k = 3, n = 9, ans = [[1,2,6], [1,3,5], [2,3,4]]`

```c++
class Solution {
public:
    vector<vector<int>>ans;
    vector<int>res;
    void dfs(int k, int n, int u) {
        if (n < 0) return; // 当前和已经超过n个，没必要往下搜了
        if (res.size() + 11 - u < k) return; // 剩余的数太少了
        if (res.size() == k && n == 0) { 
            ans.push_back(res);
            return ;
        } 
        if (u > n || u == 10) return; 
       
        // choose u
        res.push_back(u);
        dfs(k, n - u, u + 1);
        res.pop_back();
        // not choose
        dfs(k, n, u + 1);
    }
    vector<vector<int>> combinationSum3(int k, int n) {
        dfs(k, n, 1);
        return ans;
    }
};
```
---

### 17. 电话号码的字母组合
给定一个仅包含数字 `2-9 `的字符串，返回所有它能表示的字母组合。每个数字对应的字母与手机上的9键一致
`digits = "23", ans = ["ad","ae","af","bd","be","bf","cd","ce","cf"]`

#### 思路1：直接DFS
```c++
class Solution {
public:
    vector<string>hx = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    vector<string>ans;
    string res;
    void dfs(string s, int u) {
        if (u == s.size()) {
            ans.push_back(res);
            return;
        }
        int c = s[u] - '0';
        string a = hx[c];
        // 分别选择每个字母
        for (auto x : a) {
            res += x;
            dfs(s, u + 1);
            res.pop_back();
        }
    }
    vector<string> letterCombinations(string digits) {
        if(digits.size() == 0) return {};
        dfs(digits, 0);
        return ans;
    }
};
```

#### 思路2：不断往后追加
```c++
class Solution {
public:
    vector<string>hx = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    vector<string> letterCombinations(string digits) {
        if(digits.size() == 0) return {};
        vector<string>ans = {""};
        for (auto c : digits) {
            vector<string>res;
            string s = hx[c - '0'];
            for (auto pre : ans) {
                for (auto x : s) {
                    res.push_back(pre + x);
                }
            }
            ans = res;
        }
        return ans;
    }
};
```


---
### 39. 组合总和
给一个**无重复元素**的数组 `candidates` 和一个数 `target` ，找出 `candidates` 中可以使数字和为`target` 的 所有不同组合，并以列表形式返回。每个数可以重复使用。
`candidates = [2,3,6,7], target = 7`
`ans = [[2,2,3],[7]]`

####  DFS
每次两种选择，(1)继续选择当前位置的数，(2)往后挪一步
```c++
class Solution {
public:
    vector<vector<int>>ans;
    vector<int>res;
    void dfs(vector<int>& a, int target, int u) {
        if (target < 0) return;
        if (target == 0) {
            ans.push_back(res);
            return;
        }
        if (u == a.size()) return;
        // 继续选择当前数字
        res.push_back(a[u]);
        dfs(a, target - a[u], u);
        res.pop_back();
        // 考虑后面的数字
        dfs(a, target, u + 1);
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        dfs(candidates, target, 0);
        return ans;
    }
};
```
---
### 40. 组合总和 II
在数组中选择若干个数组成目标值target,求所有的方案。
难点在于不能有重复的组合。
例如：`nums = [1,1,2], target = 3, ans = [1, 2]`
如果直接`DFS`会出现两个`[1,2]`,如果一开始去重的话会导致结果不对，例如找`target=4`

**去重思路：**
如果选择不要当前数字时，直接跳过所有与之重复的数字，去下一个不同的数字位置上继续尝试。

```c++
class Solution {
public:
    vector<vector<int>>ans;
    vector<int>res;
    void dfs(vector<int>& a, int target, int u) {
        if (target < 0) return;
        if (target == 0) {
            ans.push_back(res);
            return ;
        }
        if (u == a.size()) return ;
        // 当前数字a[u]不选, 跳过所有与a[u]相同的数字
        int r = u + 1;
        while (r < a.size() && a[u] == a[r]) r++;
        dfs(a, target, r);
        // 选
        res.push_back(a[u]);
        dfs(a, target - a[u], u + 1);
        res.pop_back();
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        dfs(candidates, target, 0);
        return ans;
    }
};
```

---
### 131. 分割回文串
将 `s` 分割成一些子串，使每个子串都是回文串。返回所有可能的分割方案。
`s = "aab", ans = [["a","a","b"],["aa","b"]]`

```c++
class Solution {
public:
    vector<vector<int>>f; // f[i][j]表示s[i~j]之间的字符串是否是回文的
    vector<vector<string>>ans;
    vector<string>res;
    void dfs(string& s, int u) {
        if (u == s.size()) {
            ans.push_back(res);
            return;
        }
        for (int e = u; e < s.size(); e++) {
            if (f[u][e]) {
                res.push_back(s.substr(u, e - u + 1));
                dfs(s, e + 1);
                res.pop_back();
            }
        }
    }
    vector<vector<string>> partition(string s) {
        int n = s.size();
        // 预处理出f[i][j]， 得到s[i~j]是否是回文串
        f = vector<vector<int>>(n, vector<int>(n, 1));
        for (int i = n - 1; i >= 0; i--) 
            for (int j = i + 1; j < n; j++)
                f[i][j] = f[i + 1][j - 1] && (s[i] == s[j]);
        dfs(s, 0);
        return ans;
    }
};
```

---
### 93. 复原 IP 地址
给一个仅有数字组成的字符串，问分割为有效IP地址的所有方案
有效IP地址的条件：(1)一共4段，(2)每段的数字不超过255，(3)每段的数字不能有前导0

`s = "25525511135", ans = ["255.255.11.135","255.255.111.35"]`
```c++
class Solution {
public:
    vector<string>ans;
    vector<int>res; // 想当于每次选出来4个数
    void dfs(string &s, int u) {
        // 剪枝
        if (res.size() > 4 || res.size() + s.size() - u < 4) return;
        if (u == s.size()) {
            if (res.size() == 4) {
                string e = "";
                for(auto x : res) e += to_string(x) + ".";
                e.pop_back();
                ans.push_back(e);
            }
            return;
        }
        // 决策
        for (int i = u; i < s.size(); i++) {
            string e = s.substr(u, i - u + 1);
            if(check(e)) {
                res.push_back(stoi(e));
                dfs(s, i + 1);
                res.pop_back();
            }
        }
    }
    bool check(string s) {
        if(s.size() > 4) return 0;
        int x = stoi(s);
        if (x > 255) return 0;
        if (s[0] == '0' && s.size() > 1) return 0;
        return 1;
    }
    vector<string> restoreIpAddresses(string s) {
        dfs(s, 0);
        return ans;
    }
};
```

---
### 78. 子集
给定一个不包含重复元素的数组，返回数组的全部子集
**很经典的题目**

#### 思路1：DFS 每个数选或不选
```c++
class Solution {
public:
    vector<vector<int>>ans;
    vector<int>res;
    void dfs(vector<int>& a, int u) {
        if (u == a.size()) {
            ans.push_back(res);
            return ;
        }
        // 不选
        dfs(a, u + 1);
        // 选
        res.push_back(a[u]);
        dfs(a, u + 1);
        res.pop_back();
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(nums, 0);
        return ans;
    }
};
```

#### 思路2：二进制枚举

```c++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>>ans;
        int n = nums.size();
        //子集的数量肯定是2^n
        for(int i = 0; i < (1 << n); i++) {
            vector<int>res;
            for(int j = 0; j < n; j++) {
                if ((i >> j) & 1) res.push_back(nums[j]);
            }
            ans.push_back(res);
        }
        return ans;
    }
};
```

---
### 90. 子集 II
给定一个**包含重复**元素的数组，返回数组的全部子集.
注意解集不能包含重复的子集。

去重思路与**组合总和 II**一样
这种去重要求是有序的，即相同的数字在一起

```c++
class Solution {
public:
    vector<vector<int>>ans;
    vector<int>res;
    void dfs(vector<int>& nums, int u) {
        if (u == nums.size()) {
            ans.push_back(res);
            return;
        }
        // 选
        res.push_back(nums[u]);
        dfs(nums, u + 1);
        res.pop_back();

        // 不选的话跳过相同元素
        int j = u;
        while(j < nums.size() && nums[j] == nums[u]) j++;
        dfs(nums, j);
    }
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        dfs(nums, 0);
        return ans;
    }
};
```
---
### 491. 递增子序列
给一个**包含重复元素**的整数数组 `nums`，找出所有不同的递增子序列，递增子序列中至少有两个元素。
`[1,2,1,1]`
`ans = [[1,2],[1,1,1],[1,1]]`

难点在于去重：
因为此时相同的数字不在一起，不能用之前的去重思路了
当遇到相同元素时，有四种情况，1表示选，0表示不选
`11,10,01,00`其中情况2和情况3会导致重复
因此规定如果前面已经选了某个数，后面也必须选
换言之，也就是说只有当前的元素不等于刚刚选的元素时，才能跳过
因为只有当后面的元素大于先前的元素的才能放进去，保证了答案整体是有序的。因此上述去重思路成立。

可以用例子`[1,2,1,1,1,3,1]`思考
```c++
class Solution {
public:
    vector<vector<int>>ans;
    vector<int>res;
    void dfs(vector<int>& nums, int u) {
        if (u == nums.size()) {
            if (res.size() > 1) ans.push_back(res);
            return ;
        }
        // 选数字nums[u] 要求大于等于之前的元素
        if (res.empty() || nums[u] >= res.back()) {
            res.push_back(nums[u]);
            dfs(nums, u + 1);
            res.pop_back();
        }
        // 不选,如果当前元素等于先前的元素则必须选
        // 即对于[1,1,1] 可选方案为[001,011,111]因此实现了去重
        if (res.empty() || nums[u] != res.back()) 
            dfs(nums, u + 1);
    }
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        dfs(nums, 0);
        return ans;
    }
};
```

---
### 46. 全排列
给一个**不含重复**元素的数组，返回其全排列
`nums = [1,2,3]`
`ans = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]`

```c++
class Solution {
public:
    vector<vector<int>>ans;
    vector<int>res;
    unordered_map<int, bool>vis;
    void dfs(vector<int> &nums, int u) {
        if (u == nums.size()) {
            ans.push_back(res);
            return ;
        }
        for (auto x : nums) {
            if (vis[x]) continue;
            res.push_back(x);
            vis[x] = 1;
            dfs(nums, u + 1);
            res.pop_back();
            vis[x] = 0;
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        dfs(nums, 0);
        return ans;
    }
};
```
---


### 47. 全排列 II
给一个**可包含重复数字**的序列`nums`,返回其全排列
`nums = [1,1,2]`
`ans = [[1,1,2],[1,2,1],[2,1,1]]`

难点在于去重：首先进行排序
```c++
if (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1]) {
    continue;
}
```
上述条件保证然后每次填入的数一定是这个数所在重复数集合中「从左往右第一个未被填过的数字」
例如`[1, 1, 1]` 只能是从左往右依次选，不可能先选后面的然后之后选前面的。

```c++
class Solution {
public:
    vector<vector<int>>ans;
    vector<int>res;
    unordered_map<int, bool>vis;
    void dfs(vector<int>& nums, int u) {
        if (u == nums.size()) {
            ans.push_back(res);
            return ;
        }
        for (int i = 0; i < nums.size(); i++) {
            if (vis[i]) continue;
            if (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1]) continue;
            vis[i] = 1;
            res.push_back(nums[i]);
            dfs(nums, u + 1);
            res.pop_back();
            vis[i] = 0;
        }
    } 
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        dfs(nums, 0);
        return ans;
    }
};
```
---

### 51. N 皇后
给定一个`n`,在`n * n`的棋盘上放置`n`个皇后，使得彼此不在同一行，同一列，同一斜
```c++
class Solution {
public:
    vector<vector<string>>ans;
    unordered_map<int, int>vis; // vis[i] = j:第i列存的是第j层的皇后
    vector<int>res;// 存每一种合法的方案,每层的皇后应该放在第几列
    void dfs(int n, int u) { // 一共有多少层，当前考虑的是第几层
        if (u == n) {
            vector<string>ss;
            for (auto x :res) {
                string s(n, '.');
                s[x] = 'Q';
                ss.push_back(s);
            }
            ans.push_back(ss);
            return;
        }
        for (int i = 0; i < n; i++) { // 将当前u层的皇后放在第i列
            if (vis.count(i)) continue; // 当前列之前有人占了
            bool flag = 1; // 对角是否合适
            for (auto [col, row] : vis) {
                if (abs(row - u) == abs(col - i)) flag = 0; 
            }
            if (!flag) continue; // 对角不合适

            // 可以放
            res.push_back(i);
            vis[i] = u;
            dfs(n, u + 1);
            vis.erase(i);
            res.pop_back();
        }
    }
    vector<vector<string>> solveNQueens(int n) {
        dfs(n, 0);
        return ans;
    }
};
```

---
### 37. 解数独
解决`9*9`的数独问题

```c++
class Solution {
public:
    // row[i][j]表示第i行数字j是否存在  
    int row[9][10] = {0}, col[9][10] = {0}, cell[3][3][10] = {0}; // cell[i][j][k] 字块i,j中数字k是否存在
    typedef pair<int, int>PII;
    vector<PII>spa; // 存放空的位置
    vector<int>res; // 存放spa对应位置的结果 
    bool flag = 0; // 标记是否找到答案
    void dfs(vector<vector<char>>& board, int u) { // 当前处理的是spa[u]
        if (flag) return;
        if (u == spa.size()) {
            flag = 1;
            // 更新board
            for (int i = 0; i < spa.size(); i++) {
                auto [r, c] = spa[i];
                int num = res[i];
                board[r][c] = num + '0';
            }
            return;
        }
        auto [i, j] = spa[u]; 
        // 考虑在当前位置i,j放置元素k
        for (int k = 1; k <= 9; k++) {
            if (row[i][k] || col[j][k] || cell[i/3][j/3][k]) continue;
            // 可以放数字k
            row[i][k] = col[j][k] = cell[i/3][j/3][k] = 1;
            res.push_back(k);
            dfs(board, u + 1);
            res.pop_back();
            row[i][k] = col[j][k] = cell[i/3][j/3][k] = 0;
        }
    }
    void solveSudoku(vector<vector<char>>& board) { 
        // 处理哈希表
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char c = board[i][j];
                if (c == '.') {
                    spa.push_back({i, j});
                    continue;
                } 
                int x = c - '0';
                row[i][x] = col[j][x] = cell[i/3][j/3][x] = 1;
            }
        }
        dfs(board, 0);
    }
};
```
---