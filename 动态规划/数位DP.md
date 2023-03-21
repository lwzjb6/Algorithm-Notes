<font face= "楷体" size = 3>
---
<center><font face="楷体" size=6, color='red'> 数位DP </font> </center>

#### 记忆化搜索实现

#### 题型总结
---
1. 统计`[1, n]`中某个数字出现的次数
2. 统计`[1, n]`中满足要求的**特殊数字**的个数。eg:各个位数字均不同的数字

---

#### 题型一： 统计`[1, n]`中某个数字出现的次数

### 剑指 Offer 43. 1～n 整数中 1 出现的次数

#### 思路1： 找规律

例如，输入`12`，`1～12`这些整数中包含`1`的数字有`1、10、11`和`12`，`1`一共出现了`5`次。

核心问题：计算某一位出现`1`的次数

例如 计算十位出现`1`的次数
按照当前位的数字可分为三种情况：
(1) 当前位为`0`
(2) 当前位为`1`
(3) 当前位为`2~9`

```c++
(1) eg: 2304
十位出现`1`的范围：0010 ~ 2219
类比保险柜的密码锁，固定十位为`1`
其他位数字的可移动范围为：000~229
共计230种方案。
结论 = 高位数字 * 所在位数

(2) eg: 2314
十位出现`1`的范围：0010 ~ 2314
其他位数字的可移动范围为：000~234
共计235种方案
结论 = 高位数字 * 所在位数 + 低位数字 + 1

(3) eg = 2324
十位出现`1`的范围：0010 ~ 2319
其他位数字的可移动范围为：000~239
共计240种方案
结论 = (高位数字 + 1) * 所在位数
```

```c++
class Solution {
public:
    int countDigitOne(int n) {
        long long digit = 1, low = 0, high = n;
        int ans = 0;
        while(high) {
            int cur = high % 10;
            high /= 10;
            if(cur == 0) ans += (high * digit); 
            else if(cur == 1) ans += (high * digit + low + 1);
            else ans += (high + 1) * digit;
            low = cur * digit + low;
            digit *= 10;   
        }
        return ans;
    }
};
```

#### 思路2： 数位DP

`DFS(i, cnt1, islimit)` 表示从左向右遍历到第`i`位，`1`的个数为`cnt1`的满足要求的个数。之所以不需要用`isnum`解决前导`0`的问题，是因为仅统计`1`的个数，不影响。

```c++
class Solution {
public:
    int countDigitOne(int n) {
        string s = to_string(n);
        int m = s.size();
        int f[m][m];
        memset(f, -1, sizeof(f));

        function<int(int, int, int)>DFS = [&](int i, int cnt1, int islimit){
            if(i == m) return cnt1;

            if(!islimit && f[i][cnt1] != -1) return f[i][cnt1];

            int res = 0;
            int high = islimit ? s[i] - '0' : 9;
            for(int d = 0; d <= high; d++) {
                res += DFS(i + 1, cnt1 + (d == 1), islimit && d == s[i] - '0');
            } 
            if(!islimit) f[i][cnt1] = res;
            return res;
        };
        return DFS(0, 0, 1);
    }
};
```
类似题：`面试题 17.06. 2出现的次数`

---

### acwing 338. 计数问题
求 `a, b`之间的所有数字中` 0∼9 `分别出现的次数。

```c++
#include<bits/stdc++.h>
using namespace std;
int f[11][11];
int count(int n, int num) { // 计算1到n中num出现的次数
    string s = to_string(n);
    int m = s.size();
    memset(f, -1, sizeof(f));
        
    // 当从左到右遍历到第i位时，目标数字num出现的次数为cnt时，num出现的个数
    function<int(int, int, int, int)> DFS = [&](int i, int cnt, int islimit, int isnum) { 
        if(i == m) return cnt;
        if(!islimit && isnum && f[i][cnt] != -1) return f[i][cnt]; 
        
        int res = 0;
        // 当前位跳过，不选数字
        if(isnum == 0) res += DFS(i + 1, cnt, 0, 0);
        
        int low = (isnum) ? 0 : 1;
        int high = (islimit) ? s[i] - '0' : 9;
        for(int d = low; d <= high; d++) {
            res += DFS(i + 1, cnt + (d == num), islimit && d == (s[i] - '0'), 1);
        }
        if(!islimit && isnum) f[i][cnt] = res;
        return res;
    };
    
    return DFS(0, 0, 1, 0);
}
int main(){
    int a, b;
    while(cin >> a >> b && a && b){
        if(a > b) swap(a, b);
        for(int i = 0; i <= 9; i++) {
            cout << count(b, i) - count(a - 1, i) << " ";
        }
        cout << endl;
    }
    return 0;
}
```
---


#### 题型二：统计`[1, n]`中满足要求的特殊数字的个数。

### 2376. 统计特殊整数 [模板题]
如果一个正整数每一个数位**都互不相同**，我们称它是 特殊整数。
给你一个正整数`n` ，请你返回区间 `[1, n]` 之间特殊整数的数目。

`n = 20, ans = 19` 除了`11`以外都可以

整体实现思路：
（1）先将数字`n`转换为字符串, 便于考虑第`i`个字符应该填哪个数字.
因为`n <= 1e9` 所以字符串`n`最多有`10`位
（2）用`mask`来标记状态`i - 1`之前已经使用过的数字集合。例如`i = 2, ` `mask = 011`表示当前考虑字符串第`3`个位置应该填哪个数字，前面两次已经使用了数字1和数字2，因此本位不能再用。因为`mask`起到一个哈希表的作用。
（3）`isLimit` 表示当前是否受到了数字`n`的约束。若为真，则第 `i` 位填入的数字至多为 `s[i]`，否则可以是 `9`。例如`n = 23， i = 1` 如果`islimit = True, 则s[i] <= 3, 否则s[i] <= 9` 本位是否受到约束，取决于前一位是否达到了最大值`2`
（4）`isNum` 表示 `i` 前面的数位是否填了数字。
若为假，则当前位可以跳过（不填数字），或者要填入的数字至少为 `1`；
若为真，则必须填数字，且要填入的数字可以从 `0` 开始。
主要用于解决前导`0`的问题。

`f(i,mask,isLimit,isNum)` 表示构造从左往右第 `i` 位及其之后数位的合法方案数.

注意： 只需要在`islimit = False and isnum = True`的情况下需要记忆化`f[i][mask]`，其余情况下，`f[i][mask]`只会出现`1`次，没必要记忆化。
例如：
```c++
n = 325, i = 2； 
当遍历到： 12X时，此时islimit = False and isnum = True
因此需要记忆化f[1][00110], 假设f[1][00110] = 7，
那么今后当遍历到21X时，发现仍然是状态f[1][00110], 
因此可以不必再继续往下搜索了，直接读取记忆化的结果

当isnum = 0时，f[i][0]只会遇到一次
当islimit = 1时，f[i][mask]也只会出现一次，因为就是数字n的前i-1位
所以不需要记忆化搜索
```

```c++
class Solution {
public:
    int countSpecialNumbers(int n) {
        string s = to_string(n);
        int m = s.size();
        int f[m][1 << 10];      // 记忆化搜索思路
        memset(f, -1, sizeof(f)); // 表示均未访问到对应状态

        function<int(int, int, int, int)>DFS = [&](int i, int mask, int islimit, int isnum){
            if(i == m) return isnum; // 例子n = 1理解。
            if(islimit == 0 && isnum == 1 && f[i][mask] != -1) return f[i][mask];

            int res = 0;

            if(isnum == 0) res += DFS(i + 1, mask, 0, 0); // 如果可以跳过当前位，即当前位不填数字, 同时因为当前位没有选数字，所以下一位选的时候没有限制。
            
            int low = (isnum == 0) ? 1 : 0; // 如果之前还没有填过数字，那么当前位的下界是1. 否则可以填0
            int high = (islimit == 1) ? s[i] - '0' : 9; // 如果受限，只能填到s[i]
            for(int d = low; d <= high; d++) { 
                if((mask >> d & 1) == 0) // 没有重复的
                    res += DFS(i + 1, mask | (1 << d), islimit && d == (s[i] - '0'), 1); // 只有之前的位已经限制了，并且当前位的数字达到本位的最大值，才能限制下一位。
            }
            if (islimit == 0 && isnum == 1) f[i][mask] = res;
            return res;
        };

        return DFS(0, 0, 1, 0);
    }
};
```
时间复杂度: 状态个数 * 转移个数 = （$m2^m * 10$）

---

### 1012. 至少有 1 位重复的数字
直接求不好求，转化为：`n - `(数字没有重复的出现的整数) -> 上题模板

```c++
class Solution {
public:
    int numDupDigitsAtMostN(int n) {
        string s = to_string(n);
        int m = s.size();
        int f[m][1 << 10];
        memset(f, -1, sizeof(f));
        
        function<int(int, int, int, int)>DFS = [&](int i, int mask, int islimit, int isnum){
            if(i == m) return isnum;
            if(!islimit && isnum && f[i][mask] != -1) return f[i][mask];

            int res = 0;
            if(!isnum) res += DFS(i + 1, mask, 0, 0);

            int low = (isnum == 1) ? 0 : 1;
            int high = (islimit == 1) ?  s[i] - '0' : 9;

            for(int d = low; d <= high; d++) {
                if((mask >> d & 1) == 0)
                    res += DFS(i + 1, mask | (1 << d), islimit && d == s[i] - '0', 1);
            }
            if(!islimit && isnum) f[i][mask] = res;
            return res;

        };
        return n - DFS(0, 0, 1, 0);
    }
};
```
---



### 600. 不含连续1的非负整数
给定一个正整数`n` ，请你统计在 `[1, n]` 范围的非负整数中，有多少个整数的**二进制表示中**不存在连续的 `1` 。

```c++
class Solution {
public:
    int findIntegers(int n) {
        // 将n转化为二进制的string
        string s;
        while(n) {s += to_string(n % 2); n /= 2;}
        reverse(s.begin(), s.end());

        int m = s.size();
        int f[m][2];
        memset(f, -1, sizeof(f));

        function<int(int, int, int)>DFS = [&](int i, int pre, int islimit){
            if(i == m) return 1;
            if(!islimit && f[i][pre] != -1) return f[i][pre];

            int res = 0;
            int high = islimit ? s[i] - '0' : 1;
            for(int d = 0; d <= high; d++) {
                if(pre == 1 && d == 1) continue; // 保证不连续
                res += DFS(i + 1, (d == 1) ? 1 : 0, islimit && d == s[i] - '0');
            }
            if(!islimit) f[i][pre] = res;
            return res;
        };

        return DFS(0, 0, 1);
    }
};
```
---

### 902. 最大为 N 的数字组合
问可以用`digits`中的数字构造出`[1, n]`中多少个数字。
`digits = ["1","3","5","7"], n = 100`

```c++
class Solution {
public:
    int atMostNGivenDigitSet(vector<string>& digits, int n) {
        string s = to_string(n);
        int m = s.size();
        int f[m];
        memset(f, -1, sizeof(f));

        function<int(int, int, int)>DFS = [&](int i, int islimit, int isnum){
            if(i == m) return isnum;
            if(!islimit && isnum && f[i] != -1) return f[i];
            
            int res = 0;
            if(!isnum) res += DFS(i + 1, 0, 0); // 跳过不选
            char high = islimit ? s[i] : '9';
            for(auto c : digits) {
                if(c[0] > high) break;
                res += DFS(i + 1, islimit && c[0] == s[i], 1);
            }
            if(!islimit && isnum) f[i] = res;
            return res;
        };
        return DFS(0, 1, 0);
    }
};
```
---
