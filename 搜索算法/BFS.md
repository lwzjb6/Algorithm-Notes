<font face="楷体" size = 3>

<center><font face="楷体" size=6, color='red'> BFS </font> </center>

**适用场景：找最短路，最少变换次数等**

### 经典问题：迷宫找最短路
```
问从左上角走到右下角最少需要几步？
输入
5 5     迷宫大小
0 1 0 0 0   0表示可走，1表示障碍物
0 1 0 1 0
0 0 0 0 0
0 1 1 1 0
0 0 0 1 0
输出
8
```

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 110;
int n, m;
int a[N][N], vis[N][N];
struct node{
    int x, y;
    int step;
};
int dir[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
bool check(int x, int y) {
    if(x < 0 || x >= n || y < 0 || y >= m || vis[x][y] || a[x][y]) return 0;
    return 1;
}
int main()
{
    
    cin >> n >> m;
    for(int i = 0; i < n; i++) 
        for(int j = 0; j < m; j++)
            cin >> a[i][j];
    queue<node>q;
    q.push({0, 0, 0});
    while(q.size()) {
        auto e = q.front();
        q.pop();
        
        if(e.x == n - 1 && e.y == m - 1){
            cout << e.step << endl;
            break;
        }
        
        for(int i = 0; i < 4; i ++) {
            int nx = e.x + dir[i][0];
            int ny = e.y + dir[i][1];
            if(check(nx, ny)) {
                q.push({nx, ny, e.step + 1});
                vis[nx][ny] = 1;
            }
        }
    }
    
    return 0;
}
```
---

### 面试题13. 机器人的运动范围
机器人从`（0,0）`开始走，每次只能向4个方向移动一格，不能进入数位之和大于`k`的格子，例如：当`k=18`时，机器人不能进入方格 `[35, 38]`, 因为`3 + 5 + 3 + 8 > 18`。问最多进入遍历多少个格子。

易错点：容易想当然认为边界是一个斜线，因为如果当前在`(x, y)`, 并且`（x, y）`的数位之和等于k, 因此下一行的边界为`(x + 1, y - 1)`。但实际上`(9, 9)`的数位之和为`18`，`(9, 10)`的数位之和为`10`

#### BFS
```c++
class Solution {
public:
    typedef pair<int, int>pii;
    bool vis[110][110];
    int digit_sum(int x, int y) {
        int res = 0;
        while(x) {
            res += x % 10;
            x /= 10;
        }
        while(y) {
            res += y % 10;
            y /= 10;
        }
        return res;
    }
    int movingCount(int m, int n, int k) {
        queue<pii>q;
        q.push({0, 0});
        vis[0][0] = 1;
        int ans = 0;
        while(q.size()) {
            auto [x, y] = q.front();
            q.pop();
            ans ++;
            
            if(x + 1 < m && !vis[x + 1][y] && digit_sum(x + 1, y) <= k) {
                q.push({x + 1, y});
                vis[x + 1][y] = 1;
            }
            if(y + 1 < n && !vis[x][y + 1] && digit_sum(x, y + 1) <= k) {
                q.push({x, y + 1});
                vis[x][y + 1] = 1;
            }
        }  
        return ans;
    }
};
```
---

### 1263. 推箱子
就是经典的推箱子游戏，但不同于常规的理解。
题目要求返回的是推箱子的最少次数，而不是人移动的步数。

#### 双端队列`BFS`
为什么用双端队列`deque`？
题目要保证箱子移动的总步数`bd`最少，其次是人的移动步数`md`尽可能最优，因为需要保证上述两个元素构成的二元组`(bd, md)`在队列中保持单调性。即`bd`小的在队列的前面，`bd`相同时，按`md`排列。
因此，如果用传统队列，直接往队尾放，会破坏上述单调性。
所以采用`deque`对于每个箱子状态，如果推动箱子，那么推动次数加1，并且新的状态加入到**队列的末尾**；如果没推动箱子，那么推动次数不变，新的状态加入到**队列的头部**。

```c++
class Solution {
public:
    // 注意：题目要求返回的是箱子被推动的次数，而不是人移动的步数

    bool vis[410][410]; // 箱子和人的组合状态是否访问过，箱子一共m * n个状态，人也是m * n个状态
    int minPushBox(vector<vector<char>>& grid) {
        int n = grid.size(), m = grid[0].size();
        memset(vis, 0, sizeof(vis));
        int mx, my, bx, by; // 人和箱子的坐标
        // 找人和箱子的起始坐标
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 'S') mx = i, my = j;
                else if(grid[i][j] == 'B') bx = i, by = j;
            }
        }
        // 定义两个辅助函数
        auto f = [&](int x, int y) -> int { // 把一个坐标转化为唯一个数
            return x * m + y;
        };

        auto check = [&](int x, int y) -> bool {  // 判断一个坐标是否合法
            if(x >= 0 && x < n && y >= 0 && y < m && grid[x][y] != '#') return 1;
            else return 0;
        };

        int dir[5] = {-1, 0, 1, 0, -1};
        deque<tuple<int, int, int>>q; // (人的位置，箱子位置，当前推箱子的总步数)
        q.push_back({f(mx, my), f(bx, by), 0});
        vis[f(mx, my)][f(bx, by)] = 1;
        
        while (q.size()) {
            auto [man, box, d] = q.front();
            q.pop_front();

            mx = man / m, my = man % m;
            bx = box / m, by = box % m;

            // 终点判断
            if (grid[bx][by] == 'T') return d;

            // 状态转移, 考虑人即可
            for (int i = 0; i < 4; i++) {
                int nmx = mx + dir[i], nmy = my + dir[i + 1];
                if (!check(nmx, nmy)) continue;
                if (nmx == bx && nmy == by) { // 推动了箱子
                    int nbx = bx + dir[i], nby = by + dir[i + 1];
                    // 如果箱子的下一个位置合法，并且组合状态没有被访问过
                    if (check(nbx, nby) && !vis[f(nmx, nmy)][f(nbx, nby)]) {
                        q.push_back({f(nmx, nmy), f(nbx, nby), d + 1}); // 箱子移动次数+1
                        vis[f(nmx, nmy)][f(nbx, nby)] = 1;
                    }
                }
                else if (!vis[f(nmx, nmy)][f(bx, by)]) { // 人动，箱子不动
                    q.push_front({f(nmx, nmy), f(bx, by), d}); // 箱子没动，d不加
                    vis[f(nmx, nmy)][f(bx, by)] = 1;
                }
            }
        } 
        return -1;
    }
};
```
---

### 2258. 逃离火灾
二维迷宫，有空位，有火，有墙。每次人和火往四个方向移动，人先火后，均不能到墙。人要从左上角到右下角，问人最多可以在起点等待几秒并到达终点。

#### 二分答案 + 多源BFS
注意火的移动的更新方式

```c++
class Solution {
public:
    using pii = pair<int, int>;
    vector<int>dir = {-1, 0, 1, 0, -1};
    bool check(vector<vector<int>>& grid, int t) {
        int n = grid.size(), m = grid[0].size();
        bool fire[n][m]; // 拷贝一份新的，不能修改旧的
        memset(fire, 0, sizeof(fire));
        vector<pii>lf; // 上一时刻的火苗和人的位置集合

        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                if(grid[i][j] == 1) {
                    lf.push_back({i, j});
                    fire[i][j] = 1;
                }
            }
        }

        auto spread_fire = [&](){
            vector<pii>cf;
            for(auto &[x, y] : lf) {
                for(int i = 0; i < 4; i++) {
                    int nx = x + dir[i], ny = y + dir[i + 1];
                    if(nx >= 0 && nx < n && ny >= 0 && ny < m && !fire[nx][ny] && grid[nx][ny] == 0) {
                        fire[nx][ny] = 1;
                        cf.push_back({nx, ny});
                    }
                }
            }
            lf = move(cf);
        };

        // 先走t步
        while(t-- && lf.size()) spread_fire();
        if(fire[0][0]) return 0; // 起点着火了

        // 人开始走
        bool vis[n][m];
        memset(vis, 0, sizeof(vis));
        queue<pii>q;
        q.push({0, 0});
        vis[0][0] = 1;
        while(q.size()) {
            int len = q.size(); // 需要控制好每一轮过后，火也要传播一次
            while(len--) {
                auto [x, y] = q.front();
                q.pop();

                if(fire[x][y]) continue;
                for(int i = 0; i < 4; i++) {
                    int nx = x + dir[i], ny = y + dir[i + 1];
                    if(nx >= 0 && nx < n && ny >= 0 && ny < m && !fire[nx][ny] && grid[nx][ny] == 0 && !vis[nx][ny] ) {
                        if(nx == n - 1 && ny == m - 1) return 1; // 到终点了
                        q.push({nx, ny});
                        vis[nx][ny] = 1;
                    }
                }
            }
            spread_fire(); // 一轮结束
        }
        return 0;    
    }
    int maximumMinutes(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        int l = -1, r = m * n;
        while(l < r) {
            int mid = (l + r + 1) >> 1;
            if(check(grid, mid)) l = mid;
            else r = mid - 1;
        }
        if(l == m * n) return 1e9;
        else return l; 
    }
};
```
---


### 1162. 地图分析
`grid = [[1,0,1],[0,0,0],[1,0,1]], ans = 2`
`01`构成的网络，找到距离1最远的0，返回他们之间的距离。
或者理解为：`1`表示病毒，问多长时间能将整个网络覆盖满

#### 多源BFS
```c++
class Solution {
public:
    using pii = pair<int, int>;
    int dir[5] = {-1, 0, 1, 0, -1};
    int maxDistance(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        bool vis[n][m];
        memset(vis, 0, sizeof vis);

        queue<pii>q;
        // 所有 1 入队
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                if(grid[i][j] == 1) {
                    q.push({i, j});
                    vis[i][j] = 1;
                }
            }
        }
        if(q.size() == 0 || q.size() == n * m) return -1;

        int cnt = 0; // 最多走几步
        while(q.size()) {
            int len = q.size(); // 控制把旧数据用完
            while(len--) {
                auto [x, y] = q.front();
                q.pop();

                for(int i = 0; i < 4; i++) {
                    int nx = x + dir[i], ny = y + dir[i + 1];
                    if(nx >= 0 && nx < n && ny >= 0 && ny < m && !vis[nx][ny]) {
                        q.push({nx, ny});
                        vis[nx][ny] = 1;
                    }
                }
            }
            cnt++;
        }
        return cnt - 1;
    }
};
```
---