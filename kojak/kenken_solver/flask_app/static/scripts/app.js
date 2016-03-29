// transform cropper dataURI output to a Blob which Dropzone accepts
function dataURItoBlob(dataURI) {
    var byteString = atob(dataURI.split(',')[1]);
    var ab = new ArrayBuffer(byteString.length);
    var ia = new Uint8Array(ab);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: 'image/jpeg' });
}

// modal window template
var modalTemplate = '<div class="modal" tabindex="-1">'+
                        <!-- bootstrap modal here -->
                        '<h2>Crop as close to the border as possible</h2>'+
                        '<div class="image-container"></div>'+
                        '<div class="buttons">'+
                            '<button type="button" class="reset">Reset</button>'+
                            '<button type="button" class="crop-upload">Submit</button>'+
                        '</div>'+
                    '</div>';

// initialize dropzone
Dropzone.autoDiscover = false;
var myDropzone = new Dropzone(
    "#droparea",
    {
        autoProcessQueue: false,
        // ..your other parameters..

        url: "/uploadajax",
        paramName: "file", // The name that will be used to transfer the file
        maxFilesize: .300, // MB
        uploadMultiple: false,
        //Don't permit multiple uploads
        accept: function(file, done) {
            console.log("uploaded");
            done();
        },
        init: function() {
            this.on("addedfile", function() {
              if (this.files[1]!=null){
                this.removeFile(this.files[0]);
              }
            });
        },
        success: function(file,response) {
            console.log('success');
            console.log(response)
            document.getElementById("output-img-container").src = "data:image/jpg;base64,"+response;
        }

    }
);

// listen to thumbnail event
myDropzone.on('thumbnail', function (file) {
    // ignore files which were already cropped and re-rendered
    // to prevent infinite loop
    if (file.cropped) {
        return;
    }
    if (file.width < 50) {
        // validate width to prevent too small files to be uploaded
        // .. add some error message here
        return;
    }
    // cache filename to re-assign it to cropped file
    //var cachedFilename = file.name;
    //I dont need this since I'm changing filename

    // remove not cropped file from dropzone (we will replace it later)
    myDropzone.removeFile(file);

    // dynamically create modals to allow multiple files processing
    var $cropperModal = $(modalTemplate);
    // 'Crop and Upload/submit' button in a modal
    var $uploadCrop = $cropperModal.find('.crop-upload');

    // 'Reset' button in a modal
    var $reset = $cropperModal.find('.reset');

    var $img = $('<img />');
    // initialize FileReader which reads uploaded file
    var reader = new FileReader();
    reader.onloadend = function () {
        // add uploaded and read image to modal
        $cropperModal.find('.image-container').html($img);
        $img.attr('src', reader.result);

        // initialize cropper for uploaded image
        $img.cropper({
            aspectRatio: 1 / 1,
            autoCrop: true,
            movable: false,
            cropBoxResizable: true,
            minContainerWidth: 200
        });
    };
    // read uploaded file (triggers code above)
    reader.readAsDataURL(file);

    $cropperModal.modal('show');

    // listener for 'Crop and Upload' button in modal
    $uploadCrop.on('click', function() {
        // get cropped image data
        var blob = $img.cropper('getCroppedCanvas',{fillColor:'#ffffff'}).toDataURL();
        // transform it to Blob object
        var newFile = dataURItoBlob(blob);
        // set 'cropped to true' (so that we don't get to that listener again)
        newFile.cropped = true;
        // assign original filename
        //newFile.name = cachedFilename;
        newFile.name = 'puzzle.jpg'

        // add cropped file to dropzone
        myDropzone.addFile(newFile);
        // upload cropped file with dropzone
        myDropzone.processQueue();
        $cropperModal.modal('hide');
    });

    $reset.on('click', function() {
        $img.cropper('reset')
    });
});
