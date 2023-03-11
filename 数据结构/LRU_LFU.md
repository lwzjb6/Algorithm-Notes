<font face="楷体" size = 3>

<center><font face="楷体" size=6, color='red'> LRU </font> </center>

`LRU` 的全称是 `Least Recently Used`
最近使用的数据是有用的，因此当缓存满了后，优先删除最近不使用的数据。

核心是实现两个方法, 其时间复杂度均要求$O(1)$：
（1）`put(key, value)`: 放入键值对`[key, value]`,如果缓存已满，删除队尾久未使用的数据
（2）`get(key)`：得到`key`对应的`value`, 如果不存在返回`-1`

根据方法总结出所要求的数据结构的特点为：
（1）根据`key O(1)`找到`value`；因为要用哈希表
（2）维护数据访问的先后顺序，因为需要支持`O(1)`插入和删除，需要用到链表
（3）规定链表头表示最近访问的，链表尾表示最不经常访问的。

因为：**LRU 缓存算法的核心数据结构就是哈希链表，双向链表和哈希表的结合体。**

### 146. LRU 缓存
<1> 自定义结构体`DlinkListNode`实现双向链表
```c++
struct DlinkListNode{ // 定义双向链表
    int key, value; 
    DlinkListNode* next;
    DlinkListNode* prev;
    DlinkListNode(): key(0), value(0), prev(nullptr), next(nullptr) {}
    DlinkListNode(int _key, int _value): key(_key), value(_value),prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    unordered_map<int, DlinkListNode*>hx; // 某个key对应的链表位置的哈希表
    DlinkListNode *head, *tail; // 伪头部和尾部
    int capacity;

public:
    LRUCache(int _capacity) {
        capacity = _capacity;
        // 初始化指针指向
        head = new DlinkListNode();
        tail = new DlinkListNode();
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        if(!hx.count(key)) return -1;
        auto tar = hx[key]; // key 存在，通过hx定位链表位置
        // 将当前节点放在头节点之后，表示刚刚访问
        remove_node(tar);
        addToHead(tar);
        return tar->value; 
    }
    
    void put(int key, int value) {
        if(hx.count(key)) { //存在对应的键，更新下value和位置即可
            auto tar = hx[key];
            tar->value = value;
            remove_node(tar);
            addToHead(tar);
        }
        else { // 在头部添加一个新的并且判断是否超过容量
            DlinkListNode * newnode = new DlinkListNode(key, value);
            hx[key] = newnode;
            addToHead(newnode);
            if(hx.size() > capacity) { // 超过容量，删除一个最久的
                auto e = tail->prev;
                remove_node(e);
                hx.erase(e->key);
                delete e;
            }
        }
    }
    // 删除节点node
    void remove_node(DlinkListNode *node){ 
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }
    // 刚访问的节点放到首部
    void addToHead(DlinkListNode *node){
        node->next = head->next;
        node->next->prev = node;
        head->next = node;
        node->prev = head;
    }
};
```

<2> 用`STL`中的双向链表`list`实现：
```c++
using pii = pair<int, int>;
class LRUCache {
private:
    int capacity;
    unordered_map<int, list<pii>::iterator> hx; // 指向双向链表中每个元素的指针
    list<pii>v; // 双向链表集合，每个元素是{key, value}
public:
    LRUCache(int _capacity) {
        capacity = _capacity;
    }
    int get(int key) {
        if(!hx.count(key)) return -1;
        else {
            auto it = hx[key];
            int val = it->second;
            // 改变其在双向链表中的位置
            v.erase(it);
            v.push_front({key, val});
            hx[key] = v.begin();
            return val;
        }
    }
    void put(int key, int value) {
        if(hx.count(key)) { // 改下值并移动下位置
            auto it = hx[key];
            v.erase(it);
            v.push_front({key, value});
            hx[key] = v.begin();
        }
        else {
            if(hx.size() == capacity) { // 移除一个元素
                auto e = v.back(); // 不是指针
                int k = e.first; // key
                v.pop_back();
                hx.erase(k);
            }
            // 加入新的元素
            v.push_front({key, value});
            hx[key] = v.begin();
        }
    }
};
```
---



<font face="楷体" size = 3>

<center><font face="楷体" size=6, color='red'> LFU </font> </center>

`LFU` 的全称是 `Least Frequencyed Used`
移除**最不经常**，**使用频率最低**使用的数据。
对于出现频率都最低的两个数据，去掉最久的未使用的。

### 460. LFU 缓存

#### 双哈希表 + 双向链表(list)
(1) `unordered_map<int, list<node> :: iterator>key_table`:
通过`key`快速找到对应的`node`。之所以用指针，而不是`node`对象，是因为用指针的话可以在`O(1)`的时间完成`erase`, 否则用`remove`的时间复杂度为`O(n)`

作用：`find`:判断`key`对应的数据是否存在，通过得到的`freq`值定位频率对应的双向链表，维护最近访问的在前面。

(2)`unordered_map<int, list<node>>freq_table;`:
通过频率定位对应的双向链表。

作用：维护同一频率对象的访问先后关系。从而当存在多个最低频率对象时，快速删除最久未访问的对象。

```c++
struct node {
    int key, val, freq;
    node(int _key, int _val, int _freq) : key(_key), val(_val), freq(_freq) {}
};
class LFUCache {
private:
    int capacity, minfreq;
    unordered_map<int, list<node> :: iterator>key_table; // key对应的双向链表节点的地址, 之后用erase的时间复杂度为O(1)
    unordered_map<int, list<node>>freq_table; // 频率freq 对应的双向链表

public: 
    LFUCache(int _capacity) {
        capacity = _capacity;
        minfreq = 0;
    }
    
    int get(int key) {
        if(!key_table.count(key)) return -1;
        else {
            auto it = key_table[key]; // 迭代器
            int val = it->val, freq = it->freq;
            // 因为更新了频率，所以更新key_table 以及freq_table
            // 更新freq_table
            freq_table[freq].erase(it);
            if(freq_table[freq].size() == 0) {
                freq_table.erase(freq); // 删除对应频率的双向链表
                if(minfreq == freq) minfreq++;
            }
            freq_table[freq + 1].push_front(node(key, val, freq + 1)); // 放到对应频率的双向链表首部
            key_table[key] = freq_table[freq + 1].begin(); // begin对应元素正好是刚刚放在首部的元素
            return val; 
        }
    }
    
    void put(int key, int value) {
        if(key_table.count(key)) { // 存在的话更新下val和频率, 类似get的操作
            auto it = key_table[key];
            int freq = it->freq;
            freq_table[freq].erase(it);
            if(freq_table[freq].size() == 0) {
                freq_table.erase(freq); // 删除对应频率的双向链表
                if(minfreq == freq) minfreq++;
            }
            freq_table[freq + 1].push_front(node(key, value, freq + 1));
            key_table[key] = freq_table[freq + 1].begin();
        }
        else { // 创建新的节点，如果容量超过限制的话，删除元素
            if(key_table.size() == capacity) { // 删除对应元素， minfreq对应的双向链表中最后的元素
                auto it = freq_table[minfreq].back(); // 是对象而非指针
                key_table.erase(it.key);
                freq_table[minfreq].pop_back();
                if(freq_table[minfreq].size() == 0) freq_table.erase(minfreq);
            }
            // 创建新的节点
            minfreq = 1;
            freq_table[1].push_front(node(key, value, 1));
            key_table[key] = freq_table[1].begin();
        }
    }
};
```

 
