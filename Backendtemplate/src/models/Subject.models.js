import mongoose,{Schema} from 'mongoose';



const subjectSchema = new Schema({
    Subject_name: {
        type: String,
        required: true
    },
    teacher: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    Student_list: [{
        type: Schema.Types.ObjectId,
        ref: 'Student',
        required: true
    }],
});

module.exports = mongoose.model('Subject', subjectSchema);