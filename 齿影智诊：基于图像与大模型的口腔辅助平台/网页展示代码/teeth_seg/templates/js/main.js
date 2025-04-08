(function($) {

    "use strict";

    var App = {

        win: $(window),
        ww: window.innerWidth,
        wh: window.innerHeight,
        isMobile: /Android|webOS|iPhone|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent),


        init: function() {
            App.main_nav();
            App.entry_hover_effects();
            App.heros();
            App.popup();
            App.shortcodes();

            App.win.on('load', function() {
                $('body').waitForImages({
                    finished: function() {
                        setTimeout(function() {
                            $('.loader-mask').addClass('hide');

                            setTimeout(function() {
                                App.reveals();
                            }, 300);
                        }, 1000);
                    },
                    waitForAll: true
                });
            });

            App.win.on('resize', function() {
                App.ww = window.innerWidth;
                App.wh = window.innerHeight;

                App.heros();

                $('.custom_mfpopup .frame').css('height', (App.wh - 160) + 'px');
                $('.custom_mfpopup img').css({
                    'margin-top': ($('.custom_mfpopup .frame').height() - $('.custom_mfpopup img').height()) / 2
                });
            });

            $(document).on('mousemove', function(e) {
                $('.cursor').css({
                    left: e.clientX - 5,
                    top: e.clientY - 5
                });
            });

            $(document).on('mouseenter', 'a, .zoom-cursor', function() {
                $('.cursor').addClass('scale');
            }).on('mouseleave', 'a, .zoom-cursor', function() {
                $('.cursor').removeClass('scale');
            });
        },


        main_nav: function() {
            var nav = $('.main_nav'),
                trigger = $('.trigger');

            function toggle_nav() {
                if (nav.hasClass('open')) {
                    nav.removeClass('open');
                    nav.addClass('close');

                    nav.on('transitionend webkitTransitionEnd oTransitionEnd', function() {
                        nav.removeClass('close');
                    });

                    $('.header').removeClass('dark');
                } else {
                    nav.addClass('open');
                    $('.header').addClass('dark');
                }
            }

            trigger.on('click', toggle_nav);

            $('.main_nav .menu li:has(ul)').find('a').on('click', function() {
                var parent = $(this).parent(),
                    submenu = $(this).next('ul');

                if (submenu.is(':visible'))
                    parent.find('ul').slideUp(300);

                if (submenu.is(':hidden')) {
                    parent.siblings().find('ul').slideUp(300);
                    submenu.slideDown(300);
                }

                if (parent.children('ul').length == 0) return true;
                else return false;
            });
        },


        entry_hover_effects: function() {
            $('.entry').each(function() {
                var $this = $(this);

                $this.on('mouseenter', function() {
                    if ($this.hasClass('entry_fade')) {
                        var fade = '<div class="content">';
                        fade += '<div class="cat">' + $this.data('cat') + '</div>';
                        fade += '<div class="title">' + $this.data('title') + '</div>';
                        fade += '</div>';

                        $('.entry_fade_content').html(fade);
                        $('.entry_fade_content').addClass('visible');
                    }

                    if ($this.hasClass('entry_hover')) {
                        var hover = '<div class="cat">' + $this.data('cat') + '</div>';
                        hover += '<div class="title">' + $this.data('title') + '</div>';

                        $('.entry_hover_content').html(hover);
                        $('.entry_hover_content').addClass('visible');

                        $(document).on('mousemove', function(e) {
                            $('.entry_hover_content').css({
                                left: e.clientX - 10,
                                top: e.clientY + 25
                            });
                        });
                    }
                }).on('mouseleave', function() {
                    $('.entry_hover_content').removeClass('visible');
                    $('.entry_fade_content').removeClass('visible');

                    $('.entry').stop().animate({
                        opacity: 1
                    }, 200);
                });
            });

            $('.works_hover_list li a').on('mouseenter', function() {
                var $this = $(this),
                    thumb = $this.data('thumb');

                $('.works_hover_list .thumbs').css({
                    'background-image': 'url(' + thumb + ')',
                    'background-size': 'cover',
                    'background-position': 'center center',
                    'background-repeat': 'no-repeat'
                });

                $('.works_hover_list .thumbs').addClass('visible');

                $('.works_hover_list li a').stop().animate({
                    'opacity': 0.2
                }, 200);

                $this.stop().animate({
                    'opacity': 1
                });
            }).on('mouseleave', function() {
                $('.works_hover_list .thumbs').removeClass('visible');

                $('.works_hover_list li a').stop().animate({
                    'opacity': 1
                }, 200);
            });
        },


        heros: function() {
            $('.hero').each(function() {
                $(this).css('height', App.wh + 'px');
            });
        },


        reveals: function() {
            App.win.on('scroll', function() {
                $('.reveal').each(function(i) {
                    var $this = $(this),
                        el_top = $this.offset().top,
                        win_bottom = App.wh + App.win.scrollTop();

                    if (el_top < win_bottom) {
                        $this.delay(i * 100).queue(function() {
                            $this.addClass('reveal-in');
                        });
                    }
                });
            }).scroll();
        },


        popup: function() {
            $('.popup_link').on('click', function(e) {
                var src = $(this).attr('href'),
                    video = '',
                    title = typeof($(this).data('title')) !== "undefined" ? $(this).data('title') : '',
                    current = $('.popup_link').index($(this)),
                    total = $('.popup_link').length,
                    frame_height = App.wh - 160;

                var meta = '<div class="meta">';
                meta += '<span class="title">' + title + '</span>';
                meta += '<div class="close zoom-cursor"><span></span></div>';
                meta += '</div>';

                var frame = '<div class="frame" style="height:' + frame_height + 'px">';
                if (src.indexOf('.mp4') != -1) {
                    frame += '<div class="video"><video autoplay controls><source src="' + src + ' type="video/mp4"></video></div>';
                } else if (src.indexOf('youtube') != -1) {
                    frame += '<iframe src="' + src + '?showinfo=0&controls=0&rel=0&autoplay=1" frameborder="0"></iframe>';
                } else if (src.indexOf('vimeo') != -1) {
                    frame += '<iframe src="' + src + '?autoplay=1&byline=0&portrait=0" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>';
                } else {
                    frame += '<img src="' + src + '">';
                }
                frame += '</div>';

                var nav = '<div class="nav">';
                nav += '<div class="prev zoom-cursor"><span></span></div>';
                nav += '<div class="next zoom-cursor"><span></span></div>';
                nav += '</div>';

                $.magnificPopup.open({
                    items: {
                        src: '<div class="custom_mfpopup">' + meta + frame + nav + '</div>',
                        type: 'inline'
                    }
                });

                $('.custom_mfpopup img').css({
                    'margin-top': ($('.custom_mfpopup .frame').height() - $('.custom_mfpopup img').height()) / 2
                });

                $('.custom_mfpopup .close').on('click', function() {
                    $.magnificPopup.instance.close();
                });

                if (current == 0) {
                    $('.custom_mfpopup .nav .prev').addClass('disabled');
                }

                if (current == total - 1) {
                    $('.custom_mfpopup .nav .next').addClass('disabled');
                }

                $('.custom_mfpopup .nav .next').on('click', function() {
                    if (current < total - 1) {
                        current++;

                        var i = $('.popup_link').eq(current);
                        src = i.attr('href');
                        title = i.data('title');

                        $('.custom_mfpopup .meta .title').text(title);

                        if (src.indexOf('.mp4') != -1) {
                            video = '<div class="video"><video autoplay controls><source src="' + src + ' type="video/mp4"></video></div>';
                            $('.custom_mfpopup .frame img').remove();
                            $('.custom_mfpopup .frame').append(video);
                        } else if (src.indexOf('youtube') != -1) {
                            video = '<iframe src="' + src + '?showinfo=0&controls=0&rel=0&autoplay=1" frameborder="0"></iframe>';
                            $('.custom_mfpopup .frame img').remove();
                            $('.custom_mfpopup .frame').append(video);
                        } else if (src.indexOf('vimeo') != -1) {
                            video = '<iframe src="' + src + '?autoplay=1&byline=0&portrait=0" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>';
                            $('.custom_mfpopup .frame img').remove();
                            $('.custom_mfpopup .frame').append(video);
                        } else {
                            $('.custom_mfpopup .frame iframe').remove();
                            $('.custom_mfpopup .frame').html('<img src="' + src + '">');
                        }

                        $('.custom_mfpopup img').css({
                            'margin-top': ($('.custom_mfpopup .frame').height() - $('.custom_mfpopup img').height()) / 2
                        });

                        if (current == total - 1) {
                            $(this).addClass('disabled');
                        }

                        if (!current == 0) {
                            $('.custom_mfpopup .nav .prev').removeClass('disabled');
                        }
                    }
                });

                $('.custom_mfpopup .nav .prev').on('click', function() {
                    if (current > 0) {
                        current--;

                        var i = $('.popup_link').eq(current);
                        src = i.attr('href');
                        title = i.data('title');

                        $('.custom_mfpopup .meta .title').text(title);

                        if (src.indexOf('.mp4') != -1) {
                            video = '<div class="video"><video autoplay controls><source src="' + src + ' type="video/mp4"></video></div>';
                            $('.custom_mfpopup .frame img').remove();
                            $('.custom_mfpopup .frame').append(video);
                        } else if (src.indexOf('youtube') != -1) {
                            video = '<iframe src="' + src + '?showinfo=0&controls=0&rel=0&autoplay=1" frameborder="0"></iframe>';
                            $('.custom_mfpopup .frame img').remove();
                            $('.custom_mfpopup .frame').append(video);
                        } else if (src.indexOf('vimeo') != -1) {
                            video = '<iframe src="' + src + '?autoplay=1&byline=0&portrait=0" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>';
                            $('.custom_mfpopup .frame img').remove();
                            $('.custom_mfpopup .frame').append(video);
                        } else {
                            $('.custom_mfpopup .frame iframe').remove();
                            $('.custom_mfpopup .frame').html('<img src="' + src + '">');
                        }

                        $('.custom_mfpopup img').css({
                            'margin-top': ($('.custom_mfpopup .frame').height() - $('.custom_mfpopup img').height()) / 2
                        });

                        if (current == 0) {
                            $(this).addClass('disabled');
                        }

                        if (current < total) {
                            $('.custom_mfpopup .nav .next').removeClass('disabled');
                        }
                    }
                });

                e.preventDefault();
            });
        },


        shortcodes: function() {
            // background
            $('[data-bg]').each(function() {
                var bg = $(this).data('bg');

                $(this).css({
                    'background-image': 'url(' + bg + ')',
                    'background-size': 'cover',
                    'background-position': 'center center',
                    'background-repeat': 'no-repeat'
                });
            });

            $('[data-bg-color]').each(function() {
                var bg = $(this).data('bg-color');

                $(this).css('background-color', bg);
            });

            // magnific popup
            var popup = $('.magnific-popup');

            popup.each(function() {
                var gallery = $(this).data('gallery') == true ? 1 : 0;

                popup.magnificPopup({
                    delegate: 'a',
                    type: 'image',
                    gallery: {
                        enabled: gallery
                    }
                });
            });

            // owl slider
            $('.slider').each(function() {
                var slider = $(this),
                    dots = slider.data('dots') == true ? 1 : 0,
                    arrows = slider.data('arrows') == true ? 1 : 0,
                    items = typeof(slider.data('items')) !== "undefined" ? parseInt(slider.data('items'), 10) : 1,
                    margin = typeof(slider.data('margin')) !== "undefined" ? parseInt(slider.data('margin'), 10) : 0;

                slider.owlCarousel({
                    autoplay: true,
                    smartSpeed: 1000,
                    items: items,
                    loop: true,
                    nav: arrows,
                    dots: dots,
                    margin: margin,
                    navText: ['', '']
                });

                $('.owl-prev, .owl-next').addClass('zoom-cursor');
            });

            // back2top
            App.win.on('scroll', function() {
                var scroll = $(this).scrollTop();

                if (scroll > 100) {
                    $('.back2top').addClass('visible');
                } else {
                    $('.back2top').removeClass('visible');
                }
            });

            $('.back2top').on('click', function() {
                $('html, body').animate({
                    scrollTop: 0
                }, 300);
            });
        }
    }

    App.init();

})(jQuery);
