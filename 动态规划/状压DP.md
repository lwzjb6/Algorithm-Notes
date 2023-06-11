<font face= "楷体" size = 3>
---
<center><font face="楷体" size=6, color='red'> 状压DP </font> </center>




### 982. 按位与为零的三元组
在数组中找到`3`个下标，可以相同，求使得`nums[i] & nums[j] & nums[k] == 0` 出现的次数。

```c++
数据范围
n = nums.size <= 1000
m = nums[i] <= 2 ^ 16
```

#### 哈希 + 暴力枚举
首先两重循环枚举出`nums[i] & nums[j]`对应数值出现的次数，哈希表记录，其可能的情况最多为`m`.

```c++
class Solution {
public:
    int countTriplets(vector<int>& nums) {
        unordered_map<int, int>hx;
        for(auto x : nums) 
            for(auto y : nums) 
                hx[x & y]++;     
        int ans = 0;
        for(auto x : nums) {
           for(auto [k, v] : hx) {
               if((x & k) == 0) ans += v;
           } 
        }
        return ans;
    }
};
```
**时间复杂度$O(nm)$**

#### 状态压缩
把二进制数的表示看成集合的表示，`1`表示对应为在集合中，`0`表示对应位不在集合中。
`nums[i] & nums[j] == 0` 表示 `i`和`j`的集合表示没有交集

```c++
class Solution {
public:
    int countTriplets(vector<int>& nums) {
        unordered_map<int, int>hx;
        for(auto x : nums) 
            for(auto y : nums) 
                hx[x & y]++;
        
        int M = (1 << 16);
        int ans = 0;
        for(auto x : nums) {
            int other = (M - 1) ^ x;  // x的补集
            // 子集枚举
            int j = other; // j 就表示 不会与x出现交集的集合表示
            do {
                ans += hx[j];
                j = (j - 1) & other;
            }while(j != other);
        }
        return ans;
    }
};
```
---



### 6364. 无平方子集计数 [good]

返回数组 `nums` 中无平方子集的数目（非空）。

无平方子集定义：如果数组 `nums` 的子集中的元素乘积是一个 无平方因子数 ，则认为该子集是一个无平方子集。

数据范围： `nums.size() <= 1000. nums[i] <= 30`

思路分析：对每个数进行质因数分解，选出的子集中的元素只要不含相同的质因子就是一个合法的无平方子集。

难点：如何记录一个数的质因数分解状态：**状态压缩。**
因为`30`以内的质数为`{2,3,5,7,11,13,17,19,23,29}`
如果某个数存在某个质因子，则其对应的二进制位为`1`
例如：`10 = 2 * 5  -> mask = 101 = 5`


#### 背包
首先对所有的数进行状压表示，然后将问题转换为，对于某个体积`j`,考虑用物品装满体积`j`的方案数`f[j]`。然后`sum(f[j])`就可找到答案。

状态表示：`f[i][j]`: 表示考虑前`i`数字，体积为`j`的方案数。
状态转移：`f[i][j] = f[i - 1][j] + f[i - 1][j ^ mask]`
`mask`表示第`i`个数的状压表示，如果要加后面这一项，前提是`mask`是`j`的子集, 异或得到前`i-1`个数占据的体积。

倒序枚举体积压缩到一维。

```c++
class Solution {
public:
    using ll = long long;
    const int mod = 1e9 + 7;
    int squareFreeSubsets(vector<int>& nums) {
        vector<int>pp = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        vector<int>hx(31, 0); // 得到每个数的状态表示
        for(int x = 2; x <= 30; x++) { // x= 1的mask=0
            // 得到x的状压表示
            for(int j = 0; j < pp.size(); j++) {
                if(x % pp[j] == 0) { // pp[j]是x的一个质因数
                    hx[x] |= (1 << j);
                    if(x % (pp[j] * pp[j]) == 0) { // x本身并不是无平方数
                        hx[x] = -1;
                        break;
                    }
                }
            }
        }
        // 01背包求方案数
        int m = (1 << pp.size());
        vector<int>f(m);
        f[0] = 1; // 背包体积为0的情况，加数字1不会增大背包的体积，因此也在f[0]中
        for(auto x : nums) {
            int mask = hx[x];
            if(mask == -1) continue; // 当前物品不能选
            for(int j = m - 1; j >= mask; j--) {
                if((j | mask) == j) { // mask是j的子集
                    f[j] = (f[j] + f[j ^ mask]) % mod; // 不选 + 选
                }
            } 
        }
        ll ans = accumulate(f.begin(), f.end(), 0L) % mod - 1;  // 减去空集
        return ans;
    }
};
```
**时间复杂度$O(2^{10} N)$**

#### 状压DP
同样对所有的数进行状压表示，一共`10`个质数，相当于一共$2^{10}$种可能的状态。
状态表示：`f[i]`表示状态为`i`的方案数
状态转移：假设当前数的状压表示为：`mask`, 其出现的次数为`count(mask)`, 其补码为`other`
`oher`的所有子集表示为`{j}`
`f[j | mask] += f[j] * count(mask)`
即用当前的数更新其能更新的状态。


```c++
class Solution {
public:
    using ll = long long;
    const int mod = 1e9 + 7;
    int squareFreeSubsets(vector<int>& nums) {
        vector<int>pp = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        vector<int>hx(31, 0); // 得到每个数的状态压缩表示
        for(int x = 2; x <= 30; x++) { // x= 1的mask=0
            // 得到x的状压表示
            for(int j = 0; j < pp.size(); j++) {
                if(x % pp[j] == 0) { // pp[j]是x的一个质因数
                    hx[x] |= (1 << j);
                    if(x % (pp[j] * pp[j]) == 0) { // x本身并不是无平方数
                        hx[x] = -1;
                        break;
                    }
                }
            }
        }
        vector<int>cnt(31, 0); // 每个数字出现的次数
        int num1 = 1; //  1 可以构成的子集的个数， 2^{1出现的次数}
        for(auto x : nums) {
            if(x == 1) num1 = num1 * 2 % mod;
            else cnt[x] ++;
        }
        
        // 状压DP
        int M = (1 << pp.size()); // 一共2^10种状态
        vector<int>f(M, 0);
        f[0] = 1;
        for(int x = 2; x <= 30; x++) {
            int mask = hx[x], c = cnt[x];
            if(c == 0) continue;
            if(mask <= 0) continue; // 不能选的数以及1先不考虑
            int other = (M - 1) ^ mask; // mask的补码
            // 枚举other的子集，可以举例101理解
            int j = other;
            do {
                f[j | mask] = (ll) (f[j | mask] + (ll) f[j] * c % mod) % mod;
                j = (j - 1) & other;
            }while(j != other);
        }
        ll ans = accumulate(f.begin(), f.end(), 0L) % mod;
        ans = (ans * num1) % mod; // 每个集合都可以和由1组成的子集拼接。
        return ans - 1; // 减去空集的个数
    }
};
```
**时间复杂度$O(2^{10} \times 30)$**

---

### 1125. 最小的必要团队
公司需要很多技能，每个人会部分技能，问最少需要多少人可以满足公司要求的全部技能

`req_skills = ["java","c","c++"]`
`people = [["java"],["c"],["c","c++"]]`
`ans = [0, 2]`

#### 思路1：状态压缩 + 01背包 + 求具体方案

状态表示：`f[i][j]`: 表示考虑前`i`数字，体积为`j`的方案数。
状态转移：`f[i][j] = f[i - 1][j] + f[i - 1][j & (~mask)]`

解释：`f[i - 1][j & (~mask)]` 为啥是`j & (~mask)`而不是`j ^ (mask)`

`j & (~mask)`表示体积`j`中去掉`mask`的剩余体积
`eg: j = 110 mask = 011` 剩余体积为`110 & (100) = 100`

区别在于如果用`j^(mask)` 说明技能不能多，即`110` 的状态不能由`011` 转移过来，但实际上本题场景可以`110`可以由`011 + 100`得到，因此用`j & (~mask)`

```c++
class Solution {
public:
    using ll = long long;
    vector<int> smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people) {
        unordered_map<string, int>hx;
        int cnt = 0;
        for(auto x : req_skills) hx[x] = cnt++;

        // 计算每个人的mask
        vector<ll>masks;
        for(auto x : people) {
            ll res = 0;
            for(auto y : x) res |= (1 << hx[y]);
            masks.push_back(res);
        }
        
        // 背包
        int n = masks.size(), m = (1 << cnt) - 1;
        ll f[n + 1][m + 1];
        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for(int i = 1; i <= n; i++) {
            auto x = masks[i - 1];
            for(int j = 0; j <= m; j++){
                f[i][j] = min(f[i - 1][j], (ll) f[i - 1][j & (~x)] + 1);
            }
        }

        // 求具体方案
        vector<int>ans; 
        int curm = m;
        for(int i = n ; i >= 1; i--) {
            int x = masks[i - 1];
            if(f[i][curm] == (ll) f[i - 1][curm & (~x)] + 1) {
                curm = curm & (~x);
                ans.push_back(i - 1);
            }
        }
        return ans;
    }
};
```

#### 思路2： 状压DP
即用当前的数更新其能更新的状态。
```c++
class Solution {
public:
    using ll = long long;
    int cal(ll x) { // 计算1的个数
        int ans = 0;
        while(x) {
            x = x & (x - 1);
            ans ++;
        }
        return ans;
    }
    vector<int> smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people) {
        unordered_map<string, int>hx;
        int cnt = 0;
        for(auto x : req_skills) hx[x] = cnt++;

        // 计算每个人的mask
        vector<ll>masks;
        for(auto x : people) {
            ll res = 0;
            for(auto y : x) res |= (1 << hx[y]);
            masks.push_back(res);
        }
        
        // 状压DP
        int m = (1 << cnt) - 1;
        vector<ll>f(m + 1, INT_MAX); // 表示集合为j的情况的最少组成个数，f[j] = 101, 表示最少需要2个人，ans = [0, 2]
        f[0] = 0;
        for(int j = 0; j <= m; j++) { // 依次考虑体积
            for(int i = 0; i < masks.size(); i++) {  // 用所有的物品更新其能更新到的体积
                int x = masks[i];
                if(cal(f[j | x]) > cal(f[j]) + 1)
                    f[j | x] = f[j] | (1LL << i);  // 用 x 更新 f[j | x]
            }
        }
        
        // 求具体方案 遍历f[m], 1对应的位就是答案
        vector<int>ans;
        for(int i = 0; i < masks.size(); i++) {
            if((f[m] >> i) & 1) ans.push_back(i);
        }
        return ans;
    }
};
```
---


### acwing 91. 最短Hamilton路径
给定一个带权无向图，点从`0∼n−1`标号，求起点 `0`到终点 `n−1`的最短 `Hamilton` 路径。

`Hamilton` 路径的定义是从 `0`到 `n−1`不重不漏地经过每个点恰好一次。

状态表示：`f[i][j]:`表示“已经走过的状态为”`i`,当前在`j`的最短路径

答案： `f[(1 << n) - 1][n - 1]`
状态转移：`last_state = i ^ (1 << j)`
`f[i][j] = min(f[i][j], f[last_state][k] +  weight[k][j])`

```c++
#include<bits/stdc++.h>
using namespace std;
int mmap[20][20]; // 邻接矩阵
int dp[1 << 20][20];
int main(){
    int n;
    cin >> n;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            cin >> mmap[i][j];
        }
    }
    memset(dp, 0x3f, sizeof(dp));
    dp[1][0] = 0;
    for(int i = 1; i < 1 << 20; i++){  // 枚举走过的状态
        for(int j = 0; j < n; j++){  // 枚举当前状态表示中可能的结束点
            if((i >> j) & 1){ // 合法性判断
                int last_state = (1 << j) ^ i; 
                for(int k = 0; k < n; k++){ 
                    if((last_state >> k) & 1){ 
                        dp[i][j] = min(dp[i][j], dp[last_state][k] + mmap[k][j]);
                    }
                }
            }
        }
    }
    cout << dp[(1 << n) - 1][n - 1] << endl;
    return 0;
}
```
---


### 2172. 数组的最大与和
`nums = [1,2,3,4,5,6], numSlots = 3, ans = 9`
给一组物品，每个物品的号码为`nums[i]`, 将其放入篮子中，每个篮子中物品的数量不能超过2个。返回将 `nums` 中所有数放入 `numSlots` 个篮子中的最大**与和**。**与和**定义为每个数与它所在篮子编号的 按位与运算 结果之和。

```
可行的方案是 [1, 4] 放入篮子 1 中，[2, 6] 放入篮子 2 中，[3, 5] 放入篮子 3 中。
(1 AND 1) + (4 AND 1) + (2 AND 2) + (6 AND 2) + (3 AND 3) + (5 AND 3) = 1 + 0 + 2 + 2 + 3 + 1 = 9 。
```

#### 状压DP

```c++
class Solution {
public:
    int count_1(int x) {
        int res = 0;
        while(x) {
            x -= (x & -x);
            res++;
        }
        return res;
    }
    // 将问题转化为每个篮子变为两份，nid / 2 = oid, 选择把每个物品放到对应的篮子中
    int maximumANDSum(vector<int>& nums, int numSlots) {
        int n = nums.size();
        int m = 1 << (2 * numSlots);
        //f[i]表示将nums的前c个数字(c = i的1的个数)放到篮子中，且放了数字的篮子集合为i时的最大与和。
        //f[1011] 表示将前3个数字放到篮子中，且数字在篮子中的分布方式等于1011的最大与和
        vector<int>f(m, 0);
        int ans = 0;
        for(int i = 0; i < m; i++) { // 枚举状态
            int c = count_1(i); // 当前物品的编号
            if(c >= nums.size()) continue;
            // 计算状态的价值
            for(int j = 0; j < 2 * numSlots; j++) {
                int res = 0;
                if(((i >> j) & 1) == 0) { // 枚举空篮子j，放当前的物品c
                    int s = i | (1 << j); // 假设把物品c当到篮子j
                    f[s] = max(f[s], f[i] + (nums[c] & (j / 2 + 1))); // 篮子j对应的真实篮子编号为 j / 2 + 1
                    ans = max(ans, f[s]);
                }
            }
        }
        return ans;
    }
};
```
---


### 1879. 两个数组最小的异或值之和
将 `nums2` 中的元素重新排列，使得 **异或值之和** 最小。
两个数组`nums1, nums2`的异或值之和定义为对应位置元素求异或后的和

#### 同上题思路
```c++
class Solution {
public:
    int count_1(int x) {
        int res = 0;
        while(x) {
            x -= (x & -x);
            res++;
        }
        return res;
    }
    int minimumXORSum(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size();
        vector<int>f((1 << n), INT_MAX);

        f[0] = 0;
        for (int i = 0; i < (1 << n); i++) {
            int c = count_1(i);
            for (int j = 0; j < n; j++) {
                if (((i >> j) & 1) == 0) { // 将第c个物品放入第j位
                    int s = i | (1 << j);
                    f[s] = min(f[s], f[i] + (nums1[c] ^ nums2[j]));
                }
            }
        }
        return f[(1 << n) - 1];
    }
};
```
---

#### 题型：将集合划分为多个子集，使得子集的最大值最小

### 2305. 公平分发饼干
`cookies = [8,15,10,20,8], k = 2, ans = 31`
`最优方案是 [8,15,8] 和 [10,20]`
将`cookies`中的元素分给`k`个人，**不公平程度**定义为所有人中获得饼干的最大值，求最小的不公平程度。

(1) 首先预处理出每种分配方案的`sum`
`eg sum[1011] = cookies[3] + cookies[1] + cookies[0]`
(2) `f[i][j]:`考虑前`i`个孩子，分配的饼干集合为`j`时的最小不公平程度。
(3) 状态转移：枚举给当前用户`i`的所有的可行分配方案`s`
$f[i][j] = min\{\quad max(f[i - 1][j / s], sum[s])\}$
```
eg f[i][101] = min{ max(f[i - 1][101] + sum[000]),
                    max(f[i - 1][100] + sum[001]),
                    max(f[i - 1][001] + sum[100]),
                    max(f[i - 1][000] + sum[101])}
```

```c++
class Solution {
public:
    int distributeCookies(vector<int>& cookies, int k) {
        int n = cookies.size();
        // 预处理sum数组
        vector<int>sum(1 << n, 0);
        for (int i = 0; i < (1 << n); i++) 
            for(int j = 0; j < n; j++) 
                if(i >> j & 1) sum[i] += cookies[j];
        
        vector<vector<int>>f(k + 1, vector<int>(1 << n, 0));
        // 边界条件
        for(int j = 0; j < (1 << n); j++) f[1][j] = sum[j];

        for (int i = 2; i <= k; i++) { // 依次考虑每个人
            for (int j = 1; j < (1 << n); j++) { // 枚举分发方式
                f[i][j] = INT_MAX;
                for(int s = j; s; s = (s - 1) & j) { // 枚举当前人的分配方式s, s 必须是j的子集
                    f[i][j] = min(f[i][j], max(f[i - 1][j ^ s], sum[s]));
                }
            }
        }
        return f[k][(1 << n) - 1];
    }
};
```
---


### 1723. 完成所有工作的最短时间
`jobs = [1,2,4,7,8], k = 2, ans = 11`
将工作分给`k`个人，最小化最大工作时间

#### 和上题一样
**倒序枚举体积优化为一维**
```c++
class Solution {
public:
    int minimumTimeRequired(vector<int>& jobs, int k) {
        int n = jobs.size();
        vector<int>sum(1 << n, 0); // 预处理所有分配方案的工作时长
        for(int i = 1; i < (1 << n); i++) 
            for(int j = 0; j < n; j++)
                if(i >> j & 1) sum[i] += jobs[j];
        
        vector<int>f(sum); // 用sum初始化第一个人的情况
        for(int i = 2; i <= k; i++) {
            for(int j = (1 << n) - 1; j > 0; j--) { // 倒序枚举分配方案
                for(int s = j; s; s = (s - 1) & j) {
                    f[j] = min(f[j], max(f[j ^ s], sum[s]));
                }
            }
        }
        return f.back();
    }
};
```