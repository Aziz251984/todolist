# -*- *-
"""
   
"""
from flask import render_template, request, Blueprint, jsonify
from flask_babel import _
from flask_login import current_user, login_required

from todoism.extensions import db
from todoism.models import Item

todo_bp = Blueprint('todo', __name__)


def new_item():
    pass

@todo_bp.route('/app')
@login_required
def app():
    all_count = Item.query.with_parent(current_user).count()
    active_count = Item.query.with_parent(current_user).filter_by(done=False).count()
    completed_count = Item.query.with_parent(current_user).filter_by(done=True).count()
    return render_template('_app.html', items=current_user.items,
                           all_count=all_count, active_count=active_count, completed_count=completed_count)


@todo_bp.route('/item/<int:item_id>/edit', methods=['PUT'])
@login_required
def edit_item(item_id):
    item = Item.query.get_or_404(item_id)
    if current_user != item.author:
        return jsonify(message=_('Permission denied.')), 403

    data = request.get_json()
    if data is None or data['body'].strip() == '':
        return jsonify(message=_('Invalid item body.')), 400
    item.body = data['body']
    db.session.commit()
    return jsonify(message=_('Item updated.'))


@todo_bp.route('/item/<int:item_id>/toggle', methods=['PATCH'])
@login_required
def toggle_item(item_id):
    item = Item.query.get_or_404(item_id)
    if current_user != item.author:
        return jsonify(message=_('Permission denied.')), 403

    item.done = not item.done
    db.session.commit()
    return jsonify(message=_('Item toggled.'))


@todo_bp.route('/item/<int:item_id>/delete', methods=['DELETE'])
@login_required
def delete_item(item_id):
    item = Item.query.get_or_404(item_id)
    if current_user != item.author:
        return jsonify(message=_('Permission denied.')), 403

    db.session.delete(item)
    db.session.commit()
    return jsonify(message=_('Item deleted.'))


@todo_bp.route('/item/clear', methods=['DELETE'])
@login_required
def clear_items():
    items = Item.query.with_parent(current_user).filter_by(done=True).all()
    for item in items:
        db.session.delete(item)
    db.session.commit()
    return jsonify(message=_('All clear!'))


@todo_bp.route('/items', methods=['POST'])
@login_required
def new_item():
    """Create a new task."""
    data = request.get_json()
    body = data.get('body')
    category = data.get('category', 'General')  # Default category
    notes = data.get('notes', '')
    importance_rank = data.get('importance_rank', 1)  # Default importance
    print('here')
    print('here comes the money: ', data)

    if not body:
        return jsonify(message=_('Task description is required.')), 400

    item = Item(
        body=body,
        category=category,
        notes=notes,
        importance_rank=importance_rank,
        author=current_user._get_current_object()
    )

    db.session.add(item)
    db.session.commit()

    return jsonify(html=render_template('_item.html', item=item), message=_('Task created.'))


@todo_bp.route('/items/<int:item_id>', methods=['PUT'])
@login_required
def update_item(item_id):
    """Update an existing task."""
    item = Item.query.get_or_404(item_id)

    if item.author != current_user:
        return jsonify(message=_('Permission denied.')), 403

    data = request.get_json()
    item.body = data.get('body', item.body)
    item.category = data.get('category', item.category)
    item.notes = data.get('notes', item.notes)
    item.importance_rank = data.get('importance_rank', item.importance_rank)
    item.done = data.get('done', item.done)

    db.session.commit()

    return jsonify(message=_('Task updated.'))


@todo_bp.route('/items/<int:item_id>', methods=['GET'])
@login_required
def get_item(item_id):
    """Get a specific task."""
    item = Item.query.get_or_404(item_id)

    if item.author != current_user:
        return jsonify(message=_('Permission denied.')), 403

    return jsonify({
        'id': item.id,
        'body': item.body,
        'category': item.category,
        'notes': item.notes,
        'importance_rank': item.importance_rank,
        'done': item.done
    })


@todo_bp.route('/items', methods=['GET'])
@login_required
def get_items():
    """Get all tasks for the current user."""
    items = Item.query.filter_by(author=current_user).all()
    return jsonify([{
        'id': item.id,
        'body': item.body,
        'category': item.category,
        'notes': item.notes,
        'importance_rank': item.importance_rank,
        'done': item.done
    } for item in items])
