import flickrapi
import requests
import os


def filter_and_download(flickr, search_tags, per_page_max, pages, url_size, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for page_idx in range(pages):
        sets = flickr.photos.search(tags=search_tags, tag_mode='all', media='photos', per_page=per_page_max,
                                    page=page_idx, extras='url_n', sort='relevance')
        # sets = flickr.photos.search(text='black background', media='photos',
        #                             per_page=per_page_max, sort='interestingness-desc',
        #                             page=page_idx, extras='url_n, tags')
        photos = sets.find('photos')
        photos_iter = photos.findall('photo')
        for photo_idx, photo in enumerate(photos_iter):
            print('{}: [{:d}/{:d}] [{:d}/{:d}]'.format(search_tags, page_idx, pages, photo_idx, len(photos_iter)))
            try:
                photo_url = photo.attrib[url_size]
            except KeyError:
                continue
            # photo_tags = photo.attrib['tags']
            photo_id = photo.attrib['id']
            # r = requests.get(photo_url, verify=False)
            r = requests.get(photo_url)
            with open('{}/{}.jpg'.format(save_dir, photo_id), "wb") as code:
                code.write(r.content)


def search_meta(flickr, search_tags, per_page_max):
    sets = flickr.photos.search(tags=search_tags, tag_mode='any', media='photos', per_page=per_page_max)
    pages = int(sets.find('photos').attrib['pages'])
    total = int(sets.find('photos').attrib['total'])
    print('total find:{:d}'.format(total))
    return pages


def search_and_download(flickr, search_tags, per_page_max, pages_max, url_size, save_dir):
    # Search photos meta-data
    pages = search_meta(flickr=flickr, search_tags=search_tags, per_page_max=per_page_max)
    pages = min(pages, pages_max)
    # Search each photo in photos
    filter_and_download(flickr=flickr, search_tags=search_tags, per_page_max=per_page_max,
                        pages=pages, url_size=url_size, save_dir=save_dir)


def main():
    api_key = '0eab67de128c5d4a1260c0443e56ec5e'
    user_id = '38d031ba85e71c58'

    class_name = 'class20'
    class_style = 'dof'
    # Url size https://www.flickr.com/services/api/misc.urls.html
    url_size = 'url_n'
    # Max size: pages_max * per_page_max
    pages_max = 1
    per_page_max = 40  # max: 500
    save_dir = 'flickr/{}_{}'.format(class_name, class_style)

    # API
    flickr = flickrapi.FlickrAPI(api_key, user_id, format='etree')

    """
    Black back ground
    """
    search_tags = 'cat, dog, bird, horse, flower, car, airplane, apple, banana, orange'
    save_dir_cur = os.path.join(save_dir, class_style)
    search_and_download(flickr=flickr, search_tags=search_tags, per_page_max=per_page_max,
                        pages_max=pages_max, url_size=url_size, save_dir=save_dir_cur)

    return
    if True:
        """
        Normal
        """
        if class_name == 'all':
            search_tags = 'selective colour'
        else:
            search_tags = '{} ,-dof, close up'.format(class_name)
        save_dir_cur = os.path.join(save_dir, 'normal')
        search_and_download(flickr=flickr, search_tags=search_tags, per_page_max=per_page_max,
                            pages_max=pages_max, url_size=url_size, save_dir=save_dir_cur)
        return


if __name__ == "__main__":
    main()
