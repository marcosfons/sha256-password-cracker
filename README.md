# sha256-password-cracker

This is a `cuda` program that tries to break a list of `sha256` passwords. This
was made for the cryptography class of UFSJ university.


## The assignment

The following list was sent to us and we need to decrypt as much passwords as
possible from the given users. The format used was `username:salt:hash`, and we
knew that the sha256 was the algorithm used over the concatenation of salt +
password. And the salt used to hash was in bytes (so needs conversion between
the hexadecimal representation to bytes).

```
Admin:2609ad21084c3cc3e64f0e6777466000:87ca8c12fbc9e7686c13c6269ffce93fb66d78a002638c95edc471ed41d46f8e
user0:48cf0446f8b1977157082e2e917f6809:6f416ce900e7a39206334a28b40f609a2984332b2b5313cdafba10e2f3d6f3a5
user1:ce6923c81047dc66b5ab1dd5cbcb32a3:e0411b307c87b96c6f64776946b2d71b61f821a2641a234c86a979f1d2347bf5
user2:471603aec71c3397f56d22b2352ccd4e:94d72fe5153921c8b5ccee30e639025c7640ad15ed4c2c68e1eacb6d2db94139
user3:c72a337eff2fdd5d63d158458652e98e:0a6ab9b4100383117271cd5c7ce083be7bbb669a532cc8857356315e61340abe
user4:5d78b83c3784eb1caded1061562c1e31:a775bf388c6e99f7255169afa0769b594692d86c662f294057de91a182cb416f
user5:095a5f60a534e5378ff8f22b8022322f:7ca0ab2c0977b449e5fea2dddec83cfe5ebe039b50fb323fa7e7183546bd80f3
user6:4589331995f3a0b0b3ec55ae84b4e885:66240965684bed7ecd3ec495208364f25e964fe83aa31679f3210a5bfe32dc10
user7:50cbc93f7de01115a9d15e138215c39e:fd45ae931a6d33f2383e8b32f172c0654152a1fd8b4134de9a31b4423a15f7b5
user8:c33f2fd8ab467af03967e9a6bf66af82:c6f415b777999c168533a0a2716e6125f740235e99c03319ef0dcb1a0be06c15
user9:9d10dcdab2c1a874c85c7c028ac9702c:69017d19f71e8e34d5a53be54ca8d4d7bc9dc6c913babe3bb1e222010eba8066
userA:7475079b115f0cd181a0f35c009ef0fe:987bd44a1cc4ea16bd5cd4dbdc9d56cbc9644691b88c0ccba9aed1a14ba206e6
userB:ebe63b071dfc20dae01a8cd135572f67:12fad8a9aeb1c8ed1f988b07b32f0a9b7d7458e7c99822d1d4284bf6edcf3a3e
userC:5958d4968fc679958e32c0b22d625856:e57e3489a84e3ad626608378bde5b0873b85c8bef443998d6bb1d3f2a6c7d0bc
userD:3354623a2c1deaed1362f124c75db8a7:27a575da417e1e4cdbf4fbbe8752579b6e1d65e79731ed773a6886812e2da116
```

## Passwords specification

Each password can have a maximum of 16 of length, with upper and lower case
letters, digits and can only use the following symbols: `@`, `*`, `$` e `%`.
Considering only `ASCII` letters we have 26 upper case letters and 26 lower case
letters, 10 digits plus the 4 symbols. So the size for the character set will be
`26 + 26 + 10 + 4 = 66`. And the total number of possibilities will be:

$$
\text{{Total Possibilities}} = \sum_{n=1}^{16} (66^n)
$$

|   Max Length |            Total Possibilities |
|-------------:|-------------------------------:|
|            1 |                             66 |
|            2 |                           4422 |
|            3 |                         291918 |
|            4 |                       19266654 |
|            5 |                     1271599230 |
|            6 |                    83925549246 |
|            7 |                  5539086250302 |
|            8 |                365579692519998 |
|            9 |              24128259706319934 |
|           10 |            1592465140617115710 |
|           11 |          105102699280729636926 |
|           12 |         6936778152528156037182 |
|           13 |       457827358066858298454078 |
|           14 |     30216605632412647697969214 |
|           15 |   1994295971739234748065968190 |
|           16 | 131623534134789493372353900606 |



The number of possibilities after 10+ characters is gigantic, so to break some
password with that length is really difficult. Because of that we choose to
create the code using `CUDA` and try to utilize the full power of a graphics
card. But even using parallel execution it's impossible to test every
possibility. So was created a bunch of ways to test only likely passwords that
could be used, for that, wordlists were used.


## Authors

- Marcos Fonseca - [marcosfons](https://github.com/marcosfons)
